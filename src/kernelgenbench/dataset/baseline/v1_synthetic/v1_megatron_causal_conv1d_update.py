"""Megatron Triton causal depthwise convolution update with a linear state cache.

Official source: NVIDIA Megatron-LM ``causal_conv1d_update`` (Triton kernel).
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/ssm/ops/causal_conv1d_triton.py#L194-L287
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/ssm/ops/causal_conv1d_triton.py#L26-L192

Contract (differs from the vLLM CPU entry, which is why it is a separate item):
x is channels-last (batch, seq, dim) or (batch, dim) for one token; conv_state is
(slots, dim, state_len) with state_len >= width - 1; weight is (dim, width) with
width in {2, 3, 4}; bias is (dim,) or None.  Each token shifts the state left,
appends the token, and produces bias + sum_j weight[:, j] * window_j.
``conv_state_indices`` selects the slot per batch row and a negative slot writes
zeros and leaves the state alone.  ``intermediate_conv_states`` receives the
state after every step.  Only the string "silu" activates; the boolean True does
not.  Output dtype follows x; the cache is mutated in place.
"""
import torch


def v1_megatron_causal_conv1d_update(x, conv_state, weight, bias=None,
                                     silu_activation=None,
                                     conv_state_indices=None,
                                     intermediate_conv_states=None):
    is_2d = x.dim() == 2
    if is_2d:
        x = x.unsqueeze(1)
    batch, seq, dim = x.shape
    state_len = conv_state.shape[-1]
    width = weight.shape[-1]
    assert width in (2, 3, 4) and state_len >= width
    out = torch.empty_like(x)
    activate = silu_activation == "silu"
    indices = (list(range(batch)) if conv_state_indices is None
               else conv_state_indices.tolist())
    for b, slot in enumerate(indices):
        if slot < 0:
            out[b].zero_()
            continue
        state = conv_state[slot]
        for s in range(seq):
            # Kernel reads the last width-1 cached values before shifting and
            # appends the new token as the last tap.
            window = torch.cat((state[:, state_len - width + 1:].clone(),
                                x[b, s][:, None]), dim=1).float()
            state[:, :-1] = state[:, 1:].clone()
            state[:, -1] = x[b, s]
            acc = torch.zeros(dim, dtype=torch.float32, device=x.device) \
                if bias is None else bias.float().clone()
            for j in range(width):
                acc = acc + weight[:, j].float() * window[:, j]
            if intermediate_conv_states is not None:
                intermediate_conv_states[slot, s].copy_(state)
            if activate:
                acc = acc * torch.sigmoid(acc)
            out[b, s].copy_(acc)
    if is_2d:
        out = out.squeeze(1)
    return out
