"""SGLang vision patch block-diagonal or padded-batch boolean mask.
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/attention/vision.py#L331-L383
"""
import torch
def v1_vision_patch_attention_mask(s,cu_seqlens,flatten_batch):
    if cu_seqlens is None:return None
    cu=cu_seqlens.cpu().tolist()
    if flatten_batch:
        mask=torch.zeros((1,s,s),dtype=torch.bool)
        for a,b in zip(cu[:-1],cu[1:]):mask[...,a:b,a:b]=True
        return mask
    lens=torch.tensor([b-a for a,b in zip(cu[:-1],cu[1:])]).reshape(-1,1,1,1)
    i=torch.arange(s).reshape(1,1,1,s);j=torch.arange(s).reshape(1,1,s,1)
    return (i<lens)&(j<lens)
