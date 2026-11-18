"""SGLang DSA cache sequence length DP/CP padding.
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/attention/dsa/utils.py#L278-L293
"""
import torch
def v1_pad_dsa_cache_seqlens(dsa_cache_seqlens,global_num_tokens,attn_cp_size,attn_cp_rank,cp_interleave,cp_active,per_rank_actual_token,dp_mode,dp_slot):
    if not ((attn_cp_size>1 and cp_interleave) or global_num_tokens is not None):return dsa_cache_seqlens
    if cp_active:tokens=per_rank_actual_token[attn_cp_rank]
    else:
        if global_num_tokens is None:raise ValueError('DP token counts required outside CP-v2')
        if dp_mode=='max':tokens=max(global_num_tokens)
        elif dp_mode=='slot':tokens=global_num_tokens[dp_slot]
        else:raise ValueError('explicit DP padding mode required')
        if cp_interleave:tokens=(tokens+attn_cp_size-1)//attn_cp_size
    n=tokens-dsa_cache_seqlens.shape[0]
    return torch.cat((dsa_cache_seqlens,dsa_cache_seqlens.new_zeros((n,*dsa_cache_seqlens.shape[1:]))),0) if n>0 else dsa_cache_seqlens
