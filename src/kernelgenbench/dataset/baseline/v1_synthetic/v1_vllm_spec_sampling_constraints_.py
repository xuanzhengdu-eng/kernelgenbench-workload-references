# Source scope: vLLM pinned apply_sampling_constraints expands per-request temperature/top-k/top-p over speculative token rows, skips all-greedy, and delegates filtering to apply_top_k_top_p. Local applies equivalent native sorting/scatter in explicit row loops; its inputs/validation and backend execution are narrower.
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/rejection_sampler.py#L589-L645
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/ops/topk_topp_sampler.py#L504-L524
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/ops/topk_topp_sampler.py#L527-L568
"""Per-request speculative temperature and top-k/top-p logits constraints."""
import torch

def v1_vllm_spec_sampling_constraints_(logits,cu_num_draft_tokens,temperature,top_k,top_p,all_greedy):
    if all_greedy:return logits
    begin=0
    for r,end_tensor in enumerate(cu_num_draft_tokens):
        end=int(end_tensor);temp=float(temperature[r]);temp=1. if temp==0 else temp
        for row in range(begin,end):
            logits[row].div_(temp)
            values,idx=logits[row].sort()
            if top_k is not None:
                kth=values[values.numel()-int(top_k[r])];values.masked_fill_(values<kth,-float('inf'))
            if top_p is not None:
                mask=values.softmax(dim=0).cumsum(dim=0)<=1.-float(top_p[r]);mask[-1]=False;values.masked_fill_(mask,-float('inf'))
            logits[row].scatter_(0,idx,values)
        begin=end
    return logits
