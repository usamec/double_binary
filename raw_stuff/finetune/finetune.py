import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["OMP_NUM_THREADS"] = "1"

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datautils import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from bf16_fused_adam import BF16FusedAdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import numpy as np
import torch
import argparse
import sys
import logging

torch._inductor.config.coordinate_descent_tuning = True
#torch._logging.set_logs(dynamo = logging.INFO)

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) 
random_seed()#rank=ddp_rank)
pid = 0
print(os.sched_setaffinity(0, range(os.cpu_count())))
print(os.sched_getaffinity(pid))

import argparse
print(f"Running pytorch {torch.version.__version__}")

parser = argparse.ArgumentParser()
# file system input / output
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default="/projects/p487-24-1/pruned_nets/pruned_l1-7_0.55.pt", 
                        help="Path to the model checkpoint.")
parser.add_argument("--model_name", type=str, default="baffo32/decapoda-research-llama-7B-hf", 
                        help="Path to the model checkpoint.")
parser.add_argument("--data_path", type=str, default="/projects/p487-24-1/redpajama_tokenized_llama/",
                        help="Path to the model checkpoint.")
parser.add_argument("--base_lr", type=float, default=1e-4, 
                    help="Base learning rate.")
parser.add_argument("--q_lr_mult", type=float, default=100)
parser.add_argument("--batch_size", type=int, default=128, 
                    help="Batch size.")
parser.add_argument("--max_norm", type=float, default=20, 
                    help="Maximum norm for gradient clipping.")
parser.add_argument("--beta1", type=float, default=0.9, 
                    help="Beta1 hyperparameter for the optimizer.")
parser.add_argument("--beta2", type=float, default=0.95, 
                    help="Beta2 hyperparameter for the optimizer.")
parser.add_argument("--weight_decay", type=float, default=1e-4, 
                    help="WD")
parser.add_argument("--warmup", type=int, default=10, 
                    help="Number of warmup steps.")
parser.add_argument("--teacher_microbatch", type=int, default=32, 
                    help="Teacher microbatch size.")
parser.add_argument("--microbatch", type=int, default=8, 
                    help="Microbatch size.")
parser.add_argument("--seq_size", type=int, default=2048, 
                    help="Sequence size.")
parser.add_argument("--embed", type=int, default=4096, 
                    help="Embedding size.")
parser.add_argument("--distill", action="store_true", 
                        help="Enable distillation mode.")
parser.add_argument("--single", action="store_true", 
                        help="Not double sparse.")
parser.add_argument("--pvfrac", default=0.9999, type=float)
args = parser.parse_args()

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype='auto', attn_implementation = "flash_attention_2")
        #device_map={"model": 1, "lm_head": 0})#"sequential", max_memory={0: f'15GB', 1: '10GB'})
    print("ms", model.config.max_position_embeddings)
    model.seqlen = args.seq_size
    return model

model = get_opt(args.model_name)
model = model.cuda()

def my_pack(x):
    x = (x == 1).to(torch.uint8)
    out = torch.zeros((x.shape[0]//8), device=x.device, dtype=torch.uint8)
    for i in range(8):
        out += x[i::8] << (7 - i)
    return out

@torch.compile
def my_unpack(x):
    out = torch.zeros((x.shape[0], 8), device=x.device, dtype=torch.int8)
    for i in range(8):
        out[:,i] = (x >> (7 - i)) & 1
    return out.flatten() * 2 - 1

def power_iteration(A, num_iters=5):
    """
    Performs power iteration to compute the top singular vectors and value.

    Arguments:
        A (torch.Tensor): The input matrix of shape (m, n).
        num_iters (int): Number of iterations to perform.

    Returns:
        u (torch.Tensor): Dominant left singular vector (m,).
        sigma (torch.Tensor): Dominant singular value (scalar).
        v (torch.Tensor): Dominant right singular vector (n,).
    """
    # Start with a random vector on the appropriate device
    n = A.shape[1]
    v = torch.randn(n, device=A.device)
    v = v / torch.norm(v)

    for _ in range(num_iters):
        # Multiply A*v
        u = torch.mv(A, v)
        u_norm = torch.norm(u)
        if u_norm == 0:
            break
        u = u / u_norm

        # Multiply A^T*u
        v = torch.mv(A.t(), u)
        v_norm = torch.norm(v)
        if v_norm == 0:
            break
        v = v / v_norm

    # Estimate the dominant singular value as ||A*v||
    sigma = torch.norm(torch.mv(A, v))
    # The left singular vector corresponding to sigma:
    u = torch.mv(A, v) / sigma
    return u, sigma, v

def svd_abs(W):
    Sg = W.sign()
    Sg[Sg == 0] = 1
    u, s, v = power_iteration(W.abs(), num_iters=5)
    apx = s * torch.ger(u, v)

    return u * s, Sg, v

class BitLinear(nn.Module):
    def __init__(self, bp, shape, scale):
        super().__init__()

        self.register_buffer("bp", bp)
        self.register_parameter("scale", nn.Parameter(scale))
        self.shape = shape
        self.tuning = False

    def tune(self):
        self.tuning = True
        self.tune_steps = 0
        bit_mat = my_unpack(self.bp).reshape(self.shape).bfloat16()
        self.register_parameter("w", nn.Parameter(bit_mat))

    def untune(self):
        self.tuning = False
        del self._parameters["w"]

    def move(self):
        self.tune_steps += 1
        if self.tune_steps <= 5:
            return
        Wq = my_unpack(self.bp).reshape(self.shape).bfloat16()

        diff = (Wq - self.w.detach()) * Wq
        diff = diff.clip(0)
        thres = diff.flatten().sort()[0][int(diff.numel()*args.pvfrac)]
        mask = diff > thres
        if master_process:
            print("total change", mask.sum().item(), mask.numel(), thres.item(), diff.amax().item(), diff.amin().item())

        self.w.data[mask] = -1 * Wq[mask]
        self.bp.data = my_pack(self.w.detach().sign().flatten())

    def forward(self, x):
        bit_mat = my_unpack(self.bp).reshape(self.shape) 
        if self.tuning:
            bit_mat = bit_mat + (self.w - self.w.detach())

        return x.matmul((bit_mat*self.scale).T.to(x.dtype))

class Mul(nn.Module):
    def __init__(self, w):
        super().__init__()
        #print("w", w.amin().item(), w.median().item(), w.amax().item())

        self.register_parameter("w", nn.Parameter(w))

    def forward(self, x):
        return x * self.w.to(x.dtype)

def replace(lx, n, sd):
    out, inp = lx.weight.shape
    mid = min(inp, out)
    dev = "cuda"
    #m1 = sd[n+".0.w"].to(dev)
    #m2 = sd[n+".2.w"].to(dev)
    
    #u1, b1, v1 = svd_abs(m1.float())
    #u2, b2, v2 = svd_abs(m2.float())
    
    lx2 = nn.Sequential(
#        Mul(sd[n+".0.w"].to(dev)),
        BitLinear(sd[n+".1.bp"].to(dev), (sd[n+".2.w"].shape[0],sd[n+".0.w"].shape[0]), sd[n+".0.w"].to(dev)),
        Mul(sd[n+".2.w"].to(dev)),
        BitLinear(sd[n+".3.bp"].to(dev), (sd[n+".4.w"].shape[0],sd[n+".2.w"].shape[0]), sd[n+".4.w"].to(dev).unsqueeze(1)),
#        Mul(sd[n+".4.w"].to(dev))
    )
    return lx2

def make_d_sparse2(model):
    for i in range(32):
        model.model.layers[i].self_attn.q_proj = replace(model.model.layers[i].self_attn.q_proj, f"model.layers.{i}.self_attn.q_proj", sd)
        model.model.layers[i].self_attn.k_proj = replace(model.model.layers[i].self_attn.k_proj, f"model.layers.{i}.self_attn.k_proj", sd)
        model.model.layers[i].self_attn.v_proj = replace(model.model.layers[i].self_attn.v_proj, f"model.layers.{i}.self_attn.v_proj", sd)
        model.model.layers[i].self_attn.o_proj = replace(model.model.layers[i].self_attn.o_proj, f"model.layers.{i}.self_attn.o_proj", sd)
    
        model.model.layers[i].mlp.gate_proj = replace(model.model.layers[i].mlp.gate_proj, f"model.layers.{i}.mlp.gate_proj", sd)
        model.model.layers[i].mlp.up_proj = replace(model.model.layers[i].mlp.up_proj, f"model.layers.{i}.mlp.up_proj", sd)
        model.model.layers[i].mlp.down_proj = replace(model.model.layers[i].mlp.down_proj, f"model.layers.{i}.mlp.down_proj", sd)
#    sys.exit()
    return model
    
   
sd = torch.load(args.checkpoint, map_location="cpu")
print("check load done")
print("change to dsparse")
model = make_d_sparse2(model)
#print("reload w")
#model.load_state_dict(sd)
print("push")
model = model.bfloat16()
torch.cuda.empty_cache()
model.gradient_checkpointing_enable()
model.lm_head.weight.requires_grad = False
model.model.embed_tokens.weight.requires_grad = False
model.config.use_cache = False
#model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model

print(model)

def embed_hook(m, i, o):
    if m.training:
        o.requires_grad = True
    return o.cuda()

raw_model.model.embed_tokens.register_forward_hook(embed_hook)
raw_model.model.embed_tokens.cpu()
model.eval()

#for n, m in model.named_modules():
#    if n.endswith("proj") and np.random.randint(0, 10) == 0:
#        m[0].tune()
#        m[2].tune()

#for n, m in model.named_modules():
#    if n.endswith("proj"):
#        if m[0].tuning:
#            m[0].untune()
#        if m[2].tuning:
#            m[2].untune()




def get_opt_t(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype='auto', attn_implementation = "flash_attention_2")
    print("ms", model.config.max_position_embeddings)
    model.seqlen = args.seq_size
    return model

if args.distill:
    teacher_model = get_opt_t(args.model_name)
    teacher_model.eval()
    teacher_model = teacher_model.bfloat16()
    teacher_model.config.use_cache = False
    teacher_model = torch.compile(teacher_model.model)#, mode="max-autotune")

@torch.no_grad()
def eval_ppl_wikitext(model, testenc, bs=1*ddp_world_size, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 10 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)]
        inputs = inputs.reshape(j-i, model.seqlen)
        inputs = inputs[ddp_rank::ddp_world_size]
        if len(inputs) > 0:
    #        inputs = inputs.cuda()

            # Forward pass through the model
            lm_logits = model(inputs).logits

            # Shift logits and labels for next token prediction
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:].cuda()

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

            # Calculate negative log likelihood
            neg_log_likelihood = loss.float() * model.seqlen * len(inputs)

            #print(neg_log_likelihood)
            # Append to list of negative log likelihoods
        else:
            neg_log_likelihood = torch.Tensor([0]).cuda().sum()
        dist.all_reduce(neg_log_likelihood, op=dist.ReduceOp.SUM)
        nlls.append(neg_log_likelihood.item())
        if len(inputs) > 0:
            del inputs, lm_logits, shift_logits, shift_labels

    # Compute perplexity
    ppl = np.exp(sum(nlls) / (nsamples * model.seqlen))#torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl

print(os.sched_getaffinity(pid))
dataloader, testloader = get_loaders(
    "wikitext2", seed=0, model=args.model_name, seqlen=raw_model.seqlen
)
ppl = eval_ppl_wikitext(model, testloader)
if master_process:
    print("start ppl", ppl)
#sys.exit()
print(os.sched_getaffinity(pid))
if master_process:
    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_scalar("Val/ppl", ppl, 0)

import datasets
import transformers

dataset = datasets.load_from_disk(args.data_path)
print("dataset len", len(dataset))

model.train()
model.model = torch.compile(model.model)

#model = torch.compile(model, mode="max-autotune")
#print(model)

#to_opt1 = [p for n, p in model.model.layers.named_parameters()]# if "wqnz" not in n]
#to_opt2 = []#p for n, p in model.model.layers.named_parameters() if "wqnz" in n]
#print("to opt", len(to_opt1))#, len(to_opt2))
#opt = BF16FusedAdamW([
#opt = torch.optim.AdamW([
#    {"params": to_opt1, "lr": args.base_lr},
#    {"params": to_opt2, "lr": args.base_lr * args.q_lr_mult}#, "betas": (0, 0.95)}
#    ],
#     lr=args.base_lr, betas=(args.beta1,args.beta2), weight_decay=args.weight_decay, eps=1e-8)
#sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-13, end_factor=1, total_iters=args.warmup)

all_inds = list(range(len(dataset)))
random.shuffle(all_inds)

our_inds = all_inds[ddp_rank::ddp_world_size]

steps = len(our_inds) // args.batch_size

print(steps)
opts = {}
for n, m in model.named_modules():
    if n.endswith("proj"):
        opts[n+".0"] = BF16FusedAdamW(m[0].parameters(), lr=args.base_lr * m[0].scale.norm().item(), betas=(args.beta1,args.beta2), weight_decay=args.weight_decay, eps=1e-8)
        opts[n+".1"] = BF16FusedAdamW(m[1].parameters(), lr=args.base_lr * m[1].w.norm().item(), betas=(args.beta1,args.beta2), weight_decay=args.weight_decay, eps=1e-8)
        opts[n+".2"] = BF16FusedAdamW(m[2].parameters(), lr=args.base_lr * m[2].scale.norm().item(), betas=(args.beta1,args.beta2), weight_decay=args.weight_decay, eps=1e-8)


for s_id in range(steps):
    if s_id % 50 == 0:
        for n, m in model.named_modules():
            if n.endswith("proj"):
                if m[0].tuning:
                    m[0].untune()
                    del opts[n+".0t"]
                if m[2].tuning:
                    m[2].untune()
                    del opts[n+".2t"]

        for n, m in model.named_modules():
            if n.endswith("proj") and np.random.randint(0, 10) == 0:
                m[0].tune()
                opts[n+".0t"] = BF16FusedAdamW([m[0].w], lr=args.base_lr*args.q_lr_mult, betas=(0,args.beta2), weight_decay=args.weight_decay, eps=1e-8)
                m[2].tune()
                opts[n+".2t"] = BF16FusedAdamW([m[2].w], lr=args.base_lr*args.q_lr_mult, betas=(0,args.beta2), weight_decay=args.weight_decay, eps=1e-8)


    inds = our_inds[s_id*args.batch_size:(s_id+1)*args.batch_size]
#    inds = torch.randint(0, len(dataset), size=(args.batch_size,))
    bx = torch.LongTensor(dataset[inds]["input_ids"])
    print(bx.shape, datetime.now())
    if args.distill:
        with torch.no_grad():
            teacher_model = teacher_model.cuda()
            embeds = torch.empty(args.batch_size, args.seq_size, args.embed, dtype=torch.bfloat16, device="cuda")
            for i in range(0, args.batch_size, args.teacher_microbatch):
                embeds[i:i+args.teacher_microbatch] = teacher_model(bx[i:i+args.teacher_microbatch].cuda())[0]
            teacher_model = teacher_model.cpu()

    print("teach done", datetime.now())
#    print(embeds.shape)
    total_loss = 0
    for i in range(0, args.batch_size, args.microbatch):
        embs = model.model(bx[i:i+args.microbatch])[0]
        embs_copy = embs.detach()
        embs_copy.requires_grad = True
        for j in range(0, embs.shape[1], 1024):
            with torch.no_grad():
                probs = F.softmax(model.lm_head(embeds[i:i+args.microbatch,j:j+1024]).flatten(0, -2), dim=-1)
            our_outs = F.log_softmax(model.lm_head(embs_copy[:,j:j+1024]).flatten(0, -2), dim=-1)
#            our_outs = F.log_softmax(model(bx[i:i+args.microbatch]).logits.flatten(0, -2), dim=-1)
            loss = -(probs * (our_outs)).sum() / probs.shape[0] * args.microbatch / 8
            loss.backward()
            total_loss += loss.item()
        embs.backward(embs_copy.grad)
        del embs_copy
        del embs
        del probs
        del our_outs
        print(s_id, ddp_rank, i, loss.item(), datetime.now())

    if args.distill:
        del embeds
    for p in model.parameters():
        if p.requires_grad:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

    
#    print("step lr", opt.param_groups[0]["lr"], "norm",
#    sum(p.grad.detach().square().float().sum().cpu() if p.grad is not None else 0 for p in to_opt1 + to_opt2).sqrt().item())
    total_loss /= args.batch_size
    print(s_id, total_loss, datetime.now())
    if master_process:
        writer.add_scalar("Train/loss", total_loss / 2, s_id*args.batch_size*ddp_world_size)
#    torch.nn.utils.clip_grad_norm_(to_opt1 + to_opt2, args.max_norm)
    #for n, p in model.named_parameters():
    #    if p.grad is not None:
    #        print(n, p.norm().item(), p.grad.norm().item(), p.grad.amax().item(), p.grad.median().item())
#    if s_id > 0:
#    opt.step()
    for optx in opts.values():
        for k, v in optx.state.items():
            optx.state[k]["exp_avg"] = optx.state[k]["exp_avg"].to(k.device)
            optx.state[k]["mantissas"] = optx.state[k]["mantissas"].to(k.device)
            optx.state[k]["exp_avg_sq"] = optx.state[k]["exp_avg_sq"].to(k.device)
        optx.step()
        for k, v in optx.state.items():
            optx.state[k]["exp_avg"] = optx.state[k]["exp_avg"].cpu()
            optx.state[k]["exp_avg_sq"] = optx.state[k]["exp_avg_sq"].cpu()
            optx.state[k]["mantissas"] = optx.state[k]["mantissas"].cpu()
    for n, m in model.named_modules():
        if n.endswith("proj"):
            if m[0].tuning:
                m[0].move()
            if m[2].tuning:
                m[2].move()

    #sched.step()
    model.zero_grad(set_to_none=True)
    print("opt done", datetime.now())
    
    
    eval_every = 512 // args.batch_size
    if s_id % eval_every == eval_every - 1:
        print("Eval")
        model.eval()
        ppl = eval_ppl_wikitext(model, testloader)
        if master_process:
            writer.add_scalar("Val/ppl", ppl, s_id*args.batch_size*ddp_world_size)
            print("Wikitext perplexity:", ppl)
            torch.save(model.state_dict(), f"saves/latest_{args.run_name}.pt")
        model.train()
