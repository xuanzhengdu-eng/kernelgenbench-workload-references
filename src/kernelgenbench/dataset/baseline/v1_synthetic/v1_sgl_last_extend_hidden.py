"""SGLang final hidden row per extended sequence, identity fast path.
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/frozen_kv_mtp_utils.py#L150-L157
Input hidden [N,...] and sequence extension lengths [B]. If N == B return
the identical input; otherwise pick cumulative-length minus one. A leading
zero length selects index -1 (the last row), matching Python indexing, rather
than clamping to zero. All resulting indices must be valid for the input.
"""
import torch

def v1_sgl_last_extend_hidden(hidden_states,extend_lens):
    if hidden_states.shape[0]==len(extend_lens):return hidden_states
    lens=torch.tensor(extend_lens,device=hidden_states.device,dtype=torch.int64);ids=lens.cumsum(0)-1
    ids=torch.where(ids<0,ids+hidden_states.shape[0],ids)
    return hidden_states.index_select(0,ids)
