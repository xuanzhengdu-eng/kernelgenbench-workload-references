"""DeepSpeed unpacked SwiGLU backward: separate gate/up, tuple of gradients.

Official source (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/swiglu_triton.py#L56-L78
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/swiglu_triton.py#L80-L116
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/swiglu_triton.py#L208-L229

gate, up, and g have the same shape. The gradient is the pair
(g * up * sigmoid(gate) * (1 + gate*(1 - sigmoid(gate))), g * silu(gate)), both
cast back to the input dtype. Upstream runs the Triton backward kernel; the
verification binds the autograd Function so that kernel executes.
"""
import torch


def v1_swiglu_back_deepspeed_unpacked(g, gate, up):
    sig = torch.sigmoid(gate.float())
    ga = (g.float() * up.float() * sig * (1 + gate.float() * (1 - sig))).to(gate.dtype)
    gb = (g.float() * gate.float() * sig).to(gate.dtype)
    return ga, gb
