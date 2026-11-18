"""Pack sampled/draft IDs into input and emit logits positions.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/cpu/input_batch.cpp#L167-L224
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/input_batch.py#L475-L513
"""
import torch
def v1_vllm_combine_sampled_draft_(input_ids,idx_mapping,last_sampled_tokens,query_start_loc,seq_lens,prefill_len,draft_tokens,cu_num_logits,num_logits,num_new_sampled_tokens):
    assert num_new_sampled_tokens in (0,1)
    out=torch.empty(num_logits,dtype=torch.int64,device=input_ids.device)
    for req in range(idx_mapping.numel()):
        state=int(idx_mapping[req]);begin=int(cu_num_logits[req]);end=int(cu_num_logits[req+1]);n=end-begin;draft_n=n-num_new_sampled_tokens;query_end=int(query_start_loc[req+1]);first=query_end-n
        out[begin:end]=torch.arange(first,query_end,device=input_ids.device,dtype=torch.int64)
        seq=int(seq_lens[req]);prompt=int(prefill_len[state])
        if seq<=prompt:continue
        if num_new_sampled_tokens and seq-n>=prompt:input_ids[first]=last_sampled_tokens[state]
        if draft_n>0:input_ids[query_end-draft_n:query_end]=draft_tokens[state,:draft_n].to(input_ids.dtype)
    return out
