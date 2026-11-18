"""DeepSpeed swizzled grouped INT4/INT8 quantization, 512 FP16 elements per group.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/quantization/swizzled_quantize.cu#L20-L100
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/quantization/pt_binding.cpp#L123-L164
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/includes/quantization_utils.h#L48-L145

Input is contiguous finite FP16 and flattened; the group count must divide by
pipeline*nodes*devices.  A 512-element group keeps the kernel's thread count
exactly covering the group, so no padding lanes enter the min/max statistics.
Symmetric groups store 2*absmax/2**bits, asymmetric groups store the range
scale and the (max+min)/2 offset as float32 between the rounding halves.
Quantization rounds to nearest even in float32 and clamps to the signed range.
INT4 packs two values per byte with the even index in the high nibble. Groups
are emitted in [pipeline, device, node, local_group] order.  Returns int8
packed bytes and float32 [groups, 1 or 2] dequant scale/offset.
"""
import torch


def v1_ds_swizzle_quant(x, groups, bits, asymmetric, pipeline, nodes, devices):
    group = x.reshape(groups, 512).float()
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    if asymmetric:
        hi = group.amax(-1, keepdim=True)
        lo = group.amin(-1, keepdim=True)
        offset = (hi + lo) / 2
        inv = torch.where(hi == lo, torch.ones_like(hi), (1 << bits) / (hi - lo))
        scales = torch.cat((1 / inv, offset), dim=1)
        q = torch.round((group - offset) * inv).clamp(qmin, qmax).to(torch.int16)
    else:
        maximum = group.abs().amax(-1, keepdim=True)
        inv = torch.where(maximum == 0, torch.ones_like(maximum), (1 << bits) / (2 * maximum))
        scales = 1 / inv
        q = torch.round(group * inv).clamp(qmin, qmax).to(torch.int16)
    local = groups // (pipeline * nodes * devices)
    if bits == 4:
        packed = ((q[:, ::2] & 15) << 4) | (q[:, 1::2] & 15)
        packed = packed.to(torch.uint8).view(torch.int8)
    else:
        packed = q.to(torch.int8)
    packed = packed.reshape(nodes, devices, pipeline, local, -1).permute(2, 1, 0, 3, 4).contiguous().reshape(-1)
    scales = scales.reshape(nodes, devices, pipeline, local, -1).permute(2, 1, 0, 3, 4).contiguous().reshape(groups, -1)
    return packed, scales
