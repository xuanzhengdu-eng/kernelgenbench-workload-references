"""Move speculative accepted tree-path K/V to front of each request window.
SGLang pinned source: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/spec_utils.py#L964-L1015
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/mem_cache/memory_pool.py#L3316-L3335
Native MHA per-slot layout; cache tensors [layers,slots,heads,dim].
"""
import torch
def v1_sgl_move_accept_kv_(k_cache,v_cache,out_cache_loc,accept_index,num_correct_drafts):
    bs=accept_index.shape[0]; window=out_cache_loc.reshape(bs,-1);targets=[];sources=[]
    for i in range(bs):
        count=int(num_correct_drafts[i])+1
        targets.extend(window[i,:count].tolist())
        sources.extend(out_cache_loc[accept_index[i,:count].long()].tolist())
    tgt=torch.tensor(targets,dtype=torch.long,device=k_cache.device);src=torch.tensor(sources,dtype=torch.long,device=k_cache.device)
    k_cache[:,tgt]=k_cache[:,src].clone();v_cache[:,tgt]=v_cache[:,src].clone()
    return None
