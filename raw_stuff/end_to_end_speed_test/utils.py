import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import torch.nn.functional as F
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
from gemlite.core import GemLiteLinearTriton, DType
from transformers.cache_utils import StaticCache

from local_secrets import HF_ACCESS_TOKEN

"""
Llama loading.
"""
def get_llama(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto',
                                             local_files_only = False,
                                             token=HF_ACCESS_TOKEN,
                                             device_map="auto")
    model.seqlen = model.config.max_position_embeddings
    return model


def get_tokezizer(model_str):
    return AutoTokenizer.from_pretrained(model_str, token=HF_ACCESS_TOKEN)


def to_text(output, tokenizer):
    return tokenizer.decode(output[0], skip_special_tokens=True)

"""
Custom text generation with static graph without recompilation. 
"""

class CustomGenerator:
    def __init__(self, model, tokenizer, seq_len=128):
        model.forward = torch.compile(
             model.forward, mode="reduce-overhead", fullgraph=True)
        dev = "cuda"
        self.model = model
        self.seq_len = seq_len

        self.past_key_values = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=1 + seq_len,
            device=dev,
            dtype=torch.float16
        )
        self.cache_position = torch.arange(1, device=dev)
        
        tokenizer_input_ids = tokenizer("", return_tensors="pt").to(dev)
        self.start_input_ids = tokenizer_input_ids["input_ids"].clone()
        self.input_ids = tokenizer_input_ids["input_ids"].clone()
        self.attention_mask = tokenizer_input_ids["attention_mask"].clone()
        

    def generate(self):
        self.cache_position*=0
        self.input_ids.copy_(self.start_input_ids)
        outs = []
        for i in range(self.seq_len):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    past_key_values=self.past_key_values,
                    cache_position=self.cache_position
                )
                self.cache_position+=1
                nex_token, _ = sample(outputs.logits)
                self.input_ids.copy_(nex_token)
                outs.append(nex_token)
        torch.cuda.synchronize()
        return torch.tensor([[x for x in outs]])

"""
Gemlite related functions.
"""
GROUP_SIZE = 128

def pad_to_multiple(tensor, multiple=32, value=0):
    """
    Pads a tensor with zeroes so that each dimension is a multiple of `multiple`.
    Returns the padded tensor.
    """
    pad = []
    for dim in reversed(range(tensor.ndim)):
        size = tensor.shape[dim]
        padding_needed = (multiple - size % multiple) % multiple
        pad.extend([0, padding_needed])  # Pad after only
    padded_tensor = F.pad(tensor, pad, mode='constant', value=value)
    return padded_tensor


def my_unpack(x):
    """
    Unpacks bits succintly stored as uint8.
    Returns unpacked bits in int8.
    """
    out = torch.zeros((x.shape[0], 8), device=x.device, dtype=torch.int8)
    for i in range(8):
        out[:,i] = (x >> (7 - i)) & 1
    return out.flatten() * 2 - 1


def get_gemlite_linear(weights):
    """
    Returns gemlite linear layer for giwen weights.
    """
    W_nbits = 1
    out_features, in_features = weights.shape
    linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False, device='cpu')
    linear.weight = torch.nn.Parameter((weights).to(torch.float16), requires_grad=False)

    quant_config = BaseQuantizeConfig(nbits=W_nbits, group_size=GROUP_SIZE, quant_zero=False, quant_scale=False, axis=1)
    quant_config['weight_quant_params']['optimize'] = False

    hqq_layer = HQQLinear(linear, quant_config=quant_config, compute_dtype=torch.float16, device='cpu', del_orig=False) 
    orig_shape = (out_features, in_features)
    gemlite_linear = GemLiteLinearTriton(W_nbits=W_nbits,
                                         group_size=GROUP_SIZE, in_features=in_features, out_features=out_features,
                                         input_dtype=DType.FP16, output_dtype=DType.FP16)

    gemlite_linear.pack(hqq_layer.unpack(dtype=torch.uint8).view(orig_shape), 
                        hqq_layer.meta['scale'].clone(), hqq_layer.meta['zero'].clone(), bias=None);
    return gemlite_linear


class DoubleBinaryLinear(torch.nn.Module):
    def __init__(self, weights, layer_i, layer_name, dev="cuda"):
        super().__init__()
        full_name = f"model._orig_mod.layers.{layer_i}.{layer_name}"


        w0 = weights[full_name+".0.w"]
        w2 = weights[full_name+".2.w"]
        w4 = weights[full_name+".4.w"]

        dim_in = weights[full_name+".0.w"].shape[0]
        dim_middle = weights[full_name+".2.w"].shape[0]
        dim_out = weights[full_name+".4.w"].shape[0]
        
        bp1 = my_unpack(weights[full_name+".1.bp"]).reshape((dim_middle, dim_in))
        bp3 = my_unpack(weights[full_name+".3.bp"]).reshape((dim_out,dim_middle))
        
        bp1 = pad_to_multiple(bp1, GROUP_SIZE, 1)
        w2 = pad_to_multiple(w2, GROUP_SIZE, 0)
        bp3 = pad_to_multiple(bp3, GROUP_SIZE, 1)

        self.scaling0 = torch.nn.Parameter(w0.to(torch.float16).cuda(), requires_grad=False)
        self.binary_multiplication1 = get_gemlite_linear(bp1).cuda()
        self.scaling2 = torch.nn.Parameter(w2.to(torch.float16).cuda(), requires_grad=False)
        self.binary_multiplication3 = get_gemlite_linear(bp3).cuda()
        self.scaling4 = torch.nn.Parameter(w4.to(torch.float16).cuda(), requires_grad=False)

    def forward(self, x):
        x = (x*self.scaling0)        
        x = self.binary_multiplication1(x)
        x = (x*self.scaling2)
        x = self.binary_multiplication3(x)
        x = (x*self.scaling4)
        return x

"""
Custom generate function from https://github.com/mobiusml/hqq/blob/master/hqq/utils/generation_hf.py.

Normal transformer generate does not support proper triton kernel cuda graph optimization.
"""
def multinomial_sample_one_no_sync(probs_sort):  
    # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1,
                        keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature= 1.0, top_k= None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

@torch.compile
def sample(logits, temperature: float = 1.0, top_k= None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

