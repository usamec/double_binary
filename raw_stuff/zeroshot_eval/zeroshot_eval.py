import argparse
import json
import os
import random

import datasets
import torch
import torch.nn as nn
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
#parser.add_argument('--hf_path', default="meta-llama/Llama-2-7b-hf", type=str)
parser.add_argument('--hf_path', default="meta-llama/Meta-Llama-3-8B", type=str)
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--apply_chat_template', action='store_true')
parser.add_argument('--fewshot_as_multiturn', action='store_true')
parser.add_argument('--manifest_model', action='store_true')
parser.add_argument('--max_mem_ratio', type=float, default=0.7)
parser.add_argument('--checkpoint', type=str)

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto', attn_implementation =
    "flash_attention_2")
    print("ms", model.config.max_position_embeddings)
    model.seqlen = model.config.max_position_embeddings
    return model.cuda()

def my_pack(x):
    x = (x == 1).to(torch.uint8)
    out = torch.zeros((x.shape[0]//8), device=x.device, dtype=torch.uint8)
    for i in range(8):
        out += x[i::8] << (7 - i)
    return out

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

    
    def forward(self, x):
        bit_mat = my_unpack(self.bp).reshape(self.shape)
        return x.matmul((bit_mat*self.scale).T.to(x.dtype))
        

        
class Mul(nn.Module):
    def __init__(self, w):
        super().__init__()
        #print("w", w.amin().item(), w.median().item(), w.amax().item())
        
        self.register_buffer("w", w)
    
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
        BitLinear(sd[n+".0.bp"].to(dev), (sd[n+".1.w"].shape[0],inp), sd[n+".0.scale"].to(dev)),
        Mul(sd[n+".1.w"].to(dev)),
        BitLinear(sd[n+".2.bp"].to(dev), (out,sd[n+".1.w"].shape[0]), sd[n+".2.scale"].to(dev)),
#        Mul(sd[n+".4.w"].to(dev))
    )
    return lx2.cuda()

def make_d_sparse2(model, sd):
    for i in range(32):
        model.model.layers[i].self_attn.q_proj = replace(model.model.layers[i].self_attn.q_proj, f"model._orig_mod.layers.{i}.self_attn.q_proj", sd)
        model.model.layers[i].self_attn.k_proj = replace(model.model.layers[i].self_attn.k_proj, f"model._orig_mod.layers.{i}.self_attn.k_proj", sd)
        model.model.layers[i].self_attn.v_proj = replace(model.model.layers[i].self_attn.v_proj, f"model._orig_mod.layers.{i}.self_attn.v_proj", sd)
        model.model.layers[i].self_attn.o_proj = replace(model.model.layers[i].self_attn.o_proj, f"model._orig_mod.layers.{i}.self_attn.o_proj", sd)
    
        model.model.layers[i].mlp.gate_proj = replace(model.model.layers[i].mlp.gate_proj, f"model._orig_mod.layers.{i}.mlp.gate_proj", sd)
        model.model.layers[i].mlp.up_proj = replace(model.model.layers[i].mlp.up_proj, f"model._orig_mod.layers.{i}.mlp.up_proj", sd)
        model.model.layers[i].mlp.down_proj = replace(model.model.layers[i].mlp.down_proj, f"model._orig_mod.layers.{i}.mlp.down_proj", sd)
    return model
    
   
def main(args):
#    model, model_str = model_from_hf_path(args.hf_path, max_mem_ratio=args.max_mem_ratio, device_map='balanced')
#    print(model)
#    print(model_str)
    model_str = args.hf_path
#    model_str = "meta-llama/Llama-2-7b-hf"
    model_str = "meta-llama/Meta-Llama-3-8B"
    model = get_opt(args.hf_path) 

    # manifest for faster inference
    # use for codebooks without kernel support
    if args.manifest_model:
        for module in model.modules():
            if isinstance(module, QuantizedLinear):
                module.mode = 'train-fixW'

    tokenizer = AutoTokenizer.from_pretrained(model_str)
   

    tokenizer.pad_token = tokenizer.eos_token

    if args.checkpoint:
        sd = torch.load(args.checkpoint)
        print(sd.keys())
        model = make_d_sparse2(model, sd)
    print("model loaded")

    task_names = args.tasks.split(",")

    lm_eval_model = HFLM(model,
                         tokenizer=tokenizer,
                         batch_size=args.batch_size)

    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn)

    for key in results['results']:
        print(key)
        print()
        print(results['results'][key])
        print()
        print()

    if args.output_path is not None:
#        torch.save(results, args.output_path)
        import json
        json.dump(results["results"], open(args.output_path, "w"))


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
