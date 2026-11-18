"""DeepSpeed packed SwiGLU backward: one packed [..., 2I] tensor to [..., 2I].

Official source (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/swiglu_triton.py#L133-L153
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/swiglu_triton.py#L155-L200
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/swiglu_triton.py#L232-L252

The upstream packed kernel uses inter = width // 2 for both halves and returns a
packed gradient of the same shape. The packed 2*I contract requires an even last
width; the Megatron elementwise helper has the same even-width contract and is
the other reference in this Item.
"""
import torch


def v1_swiglu_back_deepspeed_packed(g, y):
    width = y.shape[-1]
    assert width % 2 == 0, "DeepSpeed packed SwiGLU requires an even 2*I width"
    inter = width // 2
    a = y[..., :inter]
    b = y[..., inter:]
    sig = torch.sigmoid(a.float())
    ga = (g.float() * b.float() * sig * (1 + a.float() * (1 - sig))).to(y.dtype)
    gb = (g.float() * a.float() * sig).to(y.dtype)
    return torch.cat((ga, gb), dim=-1)
