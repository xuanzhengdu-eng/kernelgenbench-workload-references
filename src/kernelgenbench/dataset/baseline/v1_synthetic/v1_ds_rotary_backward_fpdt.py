"""DeepSpeed split-half RoPE explicit backward (fpdt pure-torch entry).

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/fpdt_layer.py#L27-L39
Gradient of ``x * cos + rotate_half(x) * sin``: the rotation's transpose moves
the sine term to the other half with the opposite sign.  Rotary may cover a
prefix of the head dimension; the remaining tail passes through unchanged.
"""
import torch


def v1_ds_rotary_backward_fpdt(grad_output, freqs_cos, freqs_sin):
    n = freqs_cos.shape[-1]
    if n % 2 or n > grad_output.shape[-1]:
        raise ValueError("rotary dimension must be even and no wider than input")
    grad, tail = grad_output[..., :n], grad_output[..., n:]
    product = grad * freqs_sin
    left, right = product.chunk(2, dim=-1)
    y = grad * freqs_cos + torch.cat((right, -left), dim=-1)
    return torch.cat((y, tail), dim=-1) if tail.shape[-1] else y
