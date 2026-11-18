"""Expand one scalar per request to one scalar per speculative token.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/rejection_sampler.py#L648-L687
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/rejection_sampler.py#L1023-L1067
"""
import torch

def v1_expand_request_to_tokens(x,cu_num_tokens,num_tokens,replace_from=0,replace_to=0):
    assert x.ndim==1 and cu_num_tokens.ndim==1 and x.numel()==cu_num_tokens.numel()
    counts=torch.diff(torch.cat((cu_num_tokens.new_zeros(1),cu_num_tokens)))
    # Contract: cumulative counts are nondecreasing, sum to num_tokens, and per-request count <= 128.
    values=torch.where(x==replace_from,replace_to,x)
    return torch.repeat_interleave(values,counts.long(),output_size=num_tokens)
