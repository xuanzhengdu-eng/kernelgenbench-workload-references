"""DeepSpeed unpacked SwiGLU forward: separate gate and up tensors.

Official source (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/swiglu_triton.py#L40-L54
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/swiglu_triton.py#L208-L229

gate and up must have the same shape and dtype; the result is silu(gate) * up
with the same shape and dtype. Upstream runs the Triton forward kernel.
"""
import torch
import torch.nn.functional as F


def v1_swiglu_deepspeed_unpacked(gate, up):
    assert gate.shape == up.shape and gate.dtype == up.dtype
    return F.silu(gate) * up
