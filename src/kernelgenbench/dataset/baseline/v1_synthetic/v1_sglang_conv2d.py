"""SGLang Conv2d layer: cuDNN convolution, optional unfold+linear shortcut.

https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/conv.py#L101-L205
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/conv.py#L37-L50

``Conv2dLayer`` disables the linear shortcut by default; ``forward_cuda``
takes ``_forward_mulmat`` only when ``enable_linear`` was computed true at
construction.  ``enable_linear`` requires kernel_size == stride, zero padding,
dilation == 1 and groups == 1, so unlike vLLM a configured dilation never
disappears.  The linear branch returns a permuted, non-contiguous NCHW view;
the convolution branch returns a contiguous cuDNN result.
"""
import torch
import torch.nn.functional as F


def v1_sglang_conv2d(x, weight, bias, stride, padding, dilation, groups, linear=False):
    if linear:
        kernel = weight.shape[2:]
        if tuple(kernel) == tuple(stride) and not any(padding) and \
                all(d == 1 for d in dilation) and groups == 1:
            k1, k2 = kernel
            batches, channels, height, width = x.shape
            height, width = height // k1, width // k2
            patches = x.unfold(2, k1, k1).unfold(3, k2, k2)
            patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, channels * k1 * k2)
            out = F.linear(patches, weight.reshape(weight.shape[0], -1), bias)
            return out.view(batches, height, width, weight.shape[0]).permute(0, 3, 1, 2)
    return F.conv2d(x, weight, bias, stride=stride, padding=padding,
                    dilation=dilation, groups=groups)
