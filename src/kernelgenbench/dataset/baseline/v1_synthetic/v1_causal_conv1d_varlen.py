"""Variable-length depthwise causal convolution with mutable cache slots.

Official source: vLLM CPU implementation (the executable CPU path in the pinned
vLLM source; the native vector operator is not compiled into this x86 build).
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/mamba/ops/cpu/causal_conv1d.py#L13-L85
"""
import torch
import torch.nn.functional as F


def v1_causal_conv1d_varlen(x, weight, bias, conv_states, query_start_loc,
                            cache_indices, has_initial_state, activation):
    """Run packed sequences and update the selected cache slots in place.

    x: (channels, total_tokens), weight: (channels, width),
    bias: (channels,) or None, conv_states: (slots, channels, width-1),
    query_start_loc: (num_sequences + 1,) cumulative token offsets,
    cache_indices: (num_sequences,) or None, has_initial_state: bool vector or
    None, activation: None or "silu"/"swish".  Output: (channels, total_tokens).
    """
    out = torch.empty_like(x, dtype=conv_states.dtype)
    history = weight.shape[1] - 1
    for i in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[i])
        end = int(query_start_loc[i + 1])
        if start == end:
            continue
        slot = i if cache_indices is None else int(cache_indices[i])
        if has_initial_state is not None and bool(has_initial_state[i]):
            prefix = conv_states[slot, :, :history]
        else:
            prefix = torch.zeros_like(conv_states[slot, :, :history])
        inp = torch.cat((prefix, x[:, start:end].to(conv_states.dtype)), dim=-1).to(weight.dtype)
        value = F.conv1d(inp.unsqueeze(0), weight.unsqueeze(1), bias,
                         groups=weight.shape[0]).squeeze(0).to(conv_states.dtype)
        if activation in ("silu", "swish"):
            value = F.silu(value)
        out[:, start:end] = value
        conv_states[slot, :, :history].copy_(inp[:, -history:])
    return out.to(x.dtype)
