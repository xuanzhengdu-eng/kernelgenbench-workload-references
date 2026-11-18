"""DeepSpeed asymmetric INT4/INT8 weight dequantize + Linear (PyTorch fallback path).

Official sources (pinned commit ``8af59b2e``):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/layers.py#L64-L72
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/utils.py#L110-L154

Scope: the dequantizer branch selected when the custom CUDA dequantizer is not
eligible (``group_size % 8 != 0``, which the tests select with ``group_size=4``,
or a non-FP16 dtype / non-last ``group_dim``).  ``x`` is ``[..., H]`` and the
optional ``bias`` is ``[O]``, both in the same FP32/FP16/BF16 dtype as
``quant_scale``/``quant_min``.  Packed weights are uint8 ``[O, H]`` (8-bit) or
``[O, H/2]`` (4-bit, high nibble first).  ``group_dim`` is 0 or 1, the group
size divides that whole dimension, and the scales/minima have ``[O, H/G, 1]``
or ``[O/G, 1, H]`` shape.  Dequantization happens in the input dtype (integer
codes cast to that dtype, divided by the scale, then the minimum added) and the
result is fed to ``torch._C._nn.linear``.  A fresh tensor is returned and no
input is modified.
"""
import torch


def v1_ds_quantized_linear(x, quantized_weight, quant_scale, quant_min, bias,
                           num_bits, group_size, group_dim):
    if num_bits == 4:
        q = torch.stack((quantized_weight >> 4, quantized_weight & 15), dim=-1)
        q = q.reshape(quantized_weight.shape[0], -1)
    elif num_bits == 8:
        q = quantized_weight
    else:
        raise ValueError('bits must be 4 or 8')
    out_features, hidden = q.shape
    if group_dim == 0:
        grouped = q.reshape(out_features // group_size, group_size, hidden)
    else:
        grouped = q.reshape(out_features, hidden // group_size, group_size)
    weight = grouped.to(x.dtype).div_(quant_scale).add_(quant_min).reshape(out_features, hidden)
    return torch._C._nn.linear(x, weight, bias)
