"""vLLM pipeline-parallel sampled-token requirement, CPU metadata.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/pp_utils.py#L38-L52
"""
def v1_need_sampled_mask(old_computed,scheduled,prefill_len,max_seq_len,is_prefilling):
    need=old_computed+scheduled>=prefill_len
    if max_seq_len is not None:
        need=need & ~((prefill_len+1>=max_seq_len)&is_prefilling)
    return need if need.any() else None
