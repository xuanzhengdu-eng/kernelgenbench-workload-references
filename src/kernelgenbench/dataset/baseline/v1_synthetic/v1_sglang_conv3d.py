"""SGLang Conv3d layer: unfold+linear only for unit dilation, else cuDNN conv3d.

https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/conv.py#L207-L301
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/conv.py#L37-L50

``Conv3dLayer`` enables the shortcut unless ``disable_linear`` is passed, and
``_check_enable_linear`` requires kernel_size == stride, zero padding,
dilation == 1 and groups == 1.  A dilated convolution therefore keeps its
dilation and runs through ``F.conv3d``.  The linear branch returns a permuted,
non-contiguous NCDHW view; the convolution branch is contiguous.
"""
import torch
import torch.nn.functional as F


def v1_sglang_conv3d(x, weight, bias, stride, padding, dilation, groups, linear=True):
    if linear:
        kernel = weight.shape[2:]
        if tuple(kernel) == tuple(stride) and not any(padding) and \
                all(d == 1 for d in dilation) and groups == 1:
            k1, k2, k3 = kernel
            batches, channels, depth, height, width = x.shape
            depth, height, width = depth // k1, height // k2, width // k3
            patches = x.unfold(2, k1, k1).unfold(3, k2, k2).unfold(4, k3, k3)
            patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(
                -1, channels * k1 * k2 * k3)
            out = F.linear(patches, weight.reshape(weight.shape[0], -1), bias)
            return out.view(batches, depth, height, width, weight.shape[0]).permute(0, 4, 1, 2, 3)
    return F.conv3d(x, weight, bias, stride=stride, padding=padding,
                    dilation=dilation, groups=groups)
