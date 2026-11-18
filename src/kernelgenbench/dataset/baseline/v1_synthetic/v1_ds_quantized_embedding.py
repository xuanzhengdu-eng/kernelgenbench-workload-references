"""DeepSpeed asymmetric grouped INT4/INT8 quantized embedding.

Official sources (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):

* ``QuantizedEmbedding.forward`` -- dequantize the packed weight, then embed:
  https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/layers.py#L75-L108
* ``DeQuantizer`` -- the INT4 nibble unpack / grouped ``code / scale + min``:
  https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/utils.py#L96-L165

Scope: the forward-time dequantization + plain ``F.embedding`` with the layer's
original ``padding_idx``/``norm_type=2``/``scale_grad_by_freq=False``/``sparse=False``
arguments.  Re-quantization of a live weight (``Quantizer.quantize``) and the
``concat_to_compat_param`` packing are construction-time steps, not this entry.

Contract: ``packed`` is uint8, INT4 holds two codes per byte (high nibble first)
along the last dimension; ``quant_scale``/``quant_min`` have one value per
``group_size`` block along ``group_dim``; the decoded weight is
``codes / quant_scale + quant_min`` in ``dtype``.  ``DeQuantizer`` uses the
``quantizer`` CUDA extension when ``group_size % 8 == 0``, the group axis is the
last dimension, ``dtype`` is fp16 and the device is the accelerator; otherwise it
takes the generic torch path.  Both produce the same values.
"""
import torch
import torch.nn.functional as F


def _codes(packed, num_bits):
    if num_bits == 4:
        codes = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2),
                            device=packed.device, dtype=torch.uint8)
        codes[..., 0::2] = packed >> 4
        codes[..., 1::2] = packed & 15
        return codes
    assert num_bits == 8
    return packed


def v1_ds_quantized_embedding(indices, packed, quant_scale, quant_min, num_bits,
                              group_dim, group_size, dtype, padding_idx=None):
    """Dequantize the grouped weight, then look up ``indices``."""
    codes = _codes(packed, num_bits)
    shape = codes.shape
    groups = shape[group_dim] // group_size
    grouped = codes.view(*shape[:group_dim], groups, group_size, *shape[group_dim + 1:])
    decoded = grouped.to(dtype)
    decoded.div_(quant_scale).add_(quant_min)
    weight = decoded.view(shape)
    return F.embedding(indices, weight, padding_idx=padding_idx, max_norm=None,
                       norm_type=2, scale_grad_by_freq=False, sparse=False)
