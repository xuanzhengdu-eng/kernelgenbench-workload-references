"""vLLM Conv3d layer, CUDA dispatch: unfold+linear when the patch grid allows.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/conv.py#L16-L100
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/conv.py#L160-L211

``Conv3dLayer.forward_cuda`` deliberately replaces cuDNN with
``_forward_mulmat`` whenever ``enable_linear`` is true (kernel_size == stride,
zero padding, groups == 1) because PyTorch 2.9 disabled cuDNN Conv3D.  That
gate does not look at dilation, so a non-unit dilation is silently ignored on
this path; otherwise the layer falls back to ``F.conv3d`` with the configured
dilation.  The linear branch returns a permuted, non-contiguous NCDHW view.
"""
import torch
import torch.nn.functional as F


def v1_vllm_conv3d(x, weight, bias, stride, padding, dilation, groups):
    kernel = weight.shape[2:]
    if tuple(kernel) == tuple(stride) and not any(padding) and groups == 1:
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
