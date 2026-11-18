"""DeepSpeed asymmetric INT4/INT8 dequantize + Linear (custom CUDA dequantizer path).

Official sources (pinned commit ``8af59b2e``):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/utils.py#L110-L146
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/quantization/quantize_intX.cu#L195-L235
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/quantization/pt_binding.cpp#L83-L101

Second backend of the same ``QuantizedLinear.forward`` entry: when
``group_size % 8 == 0``, ``group_dim`` is the last dimension, the tensor dtype
is FP16 and the device is the accelerator, ``DeQuantizer.dequantize`` calls
``dequantize_int4_to_half_experimental`` / ``dequantize_int8_to_half_experimental``
instead of unpacking in Python.  The kernel converts four packed bytes (int4) or
eight bytes (int8) per thread, divides by the group scale and adds the group
minimum, all in FP16, so the dequantized weight is bit-identical to the Python
fallback.  The kernel's launch grid is ``num_group / 32`` blocks, so at least 32
groups (``O * H / group_size``) are required or the launch fails; the tests only
use eligible sizes.
"""
import torch


def v1_ds_quantized_linear_cuda(x, quantized_weight, quant_scale, quant_min, bias,
                                num_bits, group_size):
    if num_bits == 4:
        q = torch.stack((quantized_weight >> 4, quantized_weight & 15), dim=-1)
        q = q.reshape(quantized_weight.shape[0], -1)
    elif num_bits == 8:
        q = quantized_weight
    else:
        raise ValueError('bits must be 4 or 8')
    out_features, hidden = q.shape
    grouped = q.reshape(out_features, hidden // group_size, group_size)
    weight = grouped.to(x.dtype).div_(quant_scale).add_(quant_min).reshape(out_features, hidden)
    return torch._C._nn.linear(x, weight, bias)
