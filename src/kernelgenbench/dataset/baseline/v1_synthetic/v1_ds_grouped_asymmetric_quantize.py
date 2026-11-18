"""DeepSpeed per-group asymmetric INT8/packed INT4 quantization.

Source (pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/utils.py#L48-L99

``Quantizer(config).quantize(tensor)`` with ``symmetric=False``.  The tensor is
cloned, viewed as ``[..., groups, group_size, ...]`` on ``group_dim``, and each
group is linearly mapped from ``[min, max]`` to ``[0, 2**bits - 1]`` with
``scale = (2**bits - 1) / (max - min)``.  The rounded uint8 values are returned
with ``scale`` and the per-group minimum.  For 4 bits the last dimension is
packed two values per byte, high nibble first.  This reference keeps the
supported subset: ``num_bits`` in (4, 8), ``tensor.shape[group_dim]`` divisible
by ``group_size``, and an even last dimension for 4-bit packing.  The input is
not modified.
"""
import torch


def v1_ds_grouped_asymmetric_quantize(tensor, group_dim, group_size, num_bits):
    assert num_bits in (4, 8) and tensor.shape[group_dim] % group_size == 0
    shape = tensor.shape
    grouped = tensor.clone().view(
        *shape[:group_dim], shape[group_dim] // group_size, group_size,
        *shape[group_dim + 1:])
    lower = grouped.amin(dim=group_dim + 1, keepdim=True)
    upper = grouped.amax(dim=group_dim + 1, keepdim=True)
    scale = (2 ** num_bits - 1) / (upper - lower)
    shifted = (grouped - lower) * scale
    quant = torch.round(torch.clamp(shifted, 0, 2 ** num_bits - 1)).to(torch.uint8).view(shape)
    if num_bits == 4:
        assert shape[-1] % 2 == 0
        quant = (quant[..., ::2] << 4) | quant[..., 1::2]
    return quant, scale, lower
