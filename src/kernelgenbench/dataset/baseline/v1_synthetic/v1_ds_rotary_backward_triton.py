"""DeepSpeed split-half RoPE fused backward (Triton kernel).

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/fused_rotary_pos_emb.py#L64-L243
The fused kernel rotates the whole head dimension, so this branch has no
partial-rotary tail.  Every product and sum is rounded to the input dtype where
the eager expression rounds it.
"""
import torch


def v1_ds_rotary_backward_triton(grad_output, freqs_cos, freqs_sin):
    n = freqs_cos.shape[-1]
    if n % 2 or n != grad_output.shape[-1]:
        raise ValueError("fused backward rotates the whole even head dimension")
    # Upstream takes cos/sin as [batch, seq, head_dim]; a head axis is inserted
    # so they broadcast over the head dimension of [batch, heads, seq, dim].
    cos = freqs_cos.unsqueeze(1) if freqs_cos.dim() == grad_output.dim() - 1 else freqs_cos
    sin = freqs_sin.unsqueeze(1) if freqs_sin.dim() == grad_output.dim() - 1 else freqs_sin
    product = grad_output * sin
    left, right = product.chunk(2, dim=-1)
    return grad_output * cos + torch.cat((right, -left), dim=-1)
