"""DeepSpeed packed SwiGLU forward: [..., 2I] to [..., I].

Official source (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/swiglu_triton.py#L118-L131
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/swiglu_triton.py#L232-L252

inter = width // 2; gate is y[..., :inter], up is y[..., inter:2*inter], and the
result is silu(gate) * up in the input dtype. The packed 2*I contract requires an
even last width; Megatron's torch.chunk helper has the same even-width contract
and is the other reference in this Item.
"""
import torch
import torch.nn.functional as F


def v1_swiglu_deepspeed_packed(y):
    width = y.shape[-1]
    assert width % 2 == 0, "DeepSpeed packed SwiGLU requires an even 2*I width"
    inter = width // 2
    gate = y[..., :inter]
    up = y[..., inter:]
    return F.silu(gate) * up
