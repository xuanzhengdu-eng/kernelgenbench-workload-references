"""Depthwise causal convolution update with explicit in-place history state.

Official source: vLLM CPU PyTorch fallback, which is the executable CPU entry on
this x86 build (the native ``causal_conv1d_update_cpu_vec`` operator is not
compiled into the pinned vLLM extension).
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/mamba/ops/cpu/causal_conv1d.py#L119-L145
"""
import torch
import torch.nn.functional as F


def v1_causal_conv1d_update(x, conv_state, weight, bias, activation):
    """Update ``conv_state`` in place and return the new token outputs.

    x: (batch, channels, seq), conv_state: (batch, channels, state_len),
    weight: (channels, width), bias: (channels,) or None,
    activation: None or "silu"/"swish".  Output: (batch, channels, seq).
    """
    length = conv_state.shape[-1]
    seq = x.shape[-1]
    joined = torch.cat((conv_state, x), dim=-1).to(weight.dtype)
    conv_state.copy_(joined[:, :, -length:])
    out = F.conv1d(joined, weight.unsqueeze(1), bias, groups=x.shape[1])[:, :, -seq:]
    if activation in ("silu", "swish"):
        out = F.silu(out)
    return out
