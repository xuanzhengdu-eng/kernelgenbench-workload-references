"""Stable first-occurrence deduplication with fixed-shape padding (vLLM).

Official source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/backends/flex_attention.py#L266-L319

``unique_static_unsorted`` keeps the first occurrence of every value in
``[0, M]`` except ``ignored_val``, left-packs the kept values along ``dim``,
and fills the remaining slots with ``pad_val``.  Output shape and dtype match
the input and the input is not modified.  ``pad_val`` must lie in [-1, M].
"""
import torch
def v1_stable_unique_padded(x,M,dim,ignored_val,pad_val):
    if not -1<=pad_val<=M:raise ValueError('pad_val must lie in [-1, M]')
    dim=dim%x.ndim;xp=x.movedim(dim,-1);n=xp.shape[-1]
    rows=xp.reshape(-1,n);out=torch.full_like(rows,pad_val)
    for i,row in enumerate(rows.tolist()):
        seen=set();ordered=[]
        for value in row:
            if value!=ignored_val and value not in seen:seen.add(value);ordered.append(value)
        if ordered:out[i,:len(ordered)]=torch.tensor(ordered,device=x.device,dtype=x.dtype)
    return out.reshape(xp.shape).movedim(-1,dim)
