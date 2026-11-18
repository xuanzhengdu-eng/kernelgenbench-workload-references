"""vLLM Conv2d layer, CUDA dispatch: cuDNN convolution with full dilation.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/conv.py#L16-L100
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/conv.py#L110-L155

``Conv2dLayer.forward_cuda`` is the entry executed for eager inference with the
custom op enabled, and it always takes ``_forward_conv``: a plain
``F.conv2d`` with the configured stride, padding, dilation and groups.  The
unfold+linear shortcut only exists on ``forward_native``, which this commit
does not use on CUDA for 2-D convolutions, so a caller-visible
"linear output" or NHWC return does not exist for this backend.  Returns the
NCHW result.
"""
import torch.nn.functional as F


def v1_vllm_conv2d(x, weight, bias, stride, padding, dilation, groups):
    return F.conv2d(x, weight, bias, stride=stride, padding=padding,
                    dilation=dilation, groups=groups)
