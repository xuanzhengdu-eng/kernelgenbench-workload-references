"""Megatron Triton causal depthwise convolution over packed variable-length sequences.

Official source: NVIDIA Megatron-LM ``causal_conv1d_varlen_fn`` (Triton) and its
reference routine ``_causal_conv1d_varlen_simple``.
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/ssm/ops/causal_conv1d_varlen.py#L119-L208
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/ssm/ops/causal_conv1d_varlen.py#L211-L259

Contract (differs from the vLLM CPU packed entry, which mutates cache slots):
x is channels-last (total_tokens, conv_dim) and contiguous; weight is
(conv_dim, d_conv); bias is (conv_dim,); cu_seqlens is (num_requests + 1,);
initial_states is (num_requests, conv_dim, d_conv - 1) or None and is only read,
never written.  Only ``activation == "silu"`` is accepted.  Zero-length requests
are skipped.  The result is a new (total_tokens, conv_dim) tensor.
"""
import torch
import torch.nn.functional as F


def v1_megatron_causal_conv1d_varlen(x, weight, bias, cu_seqlens,
                                     initial_states=None, activation="silu"):
    assert activation == "silu", f"only silu is supported, got {activation}"
    assert x.is_contiguous() and weight.is_contiguous()
    total_tokens, conv_dim = x.shape
    width = weight.shape[1]
    requests = cu_seqlens.shape[0] - 1
    out = torch.empty_like(x)
    for r in range(requests):
        start = int(cu_seqlens[r])
        end = int(cu_seqlens[r + 1])
        if start == end:
            continue
        for t in range(end - start):
            acc = bias.float().clone()
            for j in range(width):
                src = t - (width - 1) + j
                if src < 0:
                    column = (width - 1) + src
                    tap = (initial_states[r][:, column].float()
                           if initial_states is not None and 0 <= column < width - 1
                           else torch.zeros(conv_dim, dtype=torch.float32, device=x.device))
                else:
                    tap = x[start + src].float()
                acc = acc + tap * weight[:, j].float()
            out[start + t] = (acc * torch.sigmoid(acc)).to(out.dtype)
    return out
