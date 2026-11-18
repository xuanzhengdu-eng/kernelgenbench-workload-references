"""vLLM gather speculative draft tokens and positions for logits rows.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py#L78-L145
"""
import torch

def v1_vllm_gather_draft_sampled(input_ids,positions,logits_indices,expanded_idx_mapping,expanded_local_pos,prefill_len):
    ids=input_ids[logits_indices.long()];pos=positions[logits_indices.long()];prefix=prefill_len[expanded_idx_mapping.long()];placeholder=(expanded_local_pos>0)&(pos-expanded_local_pos<prefix)
    return torch.where(placeholder,torch.full_like(ids,-1),ids),pos.clone()
