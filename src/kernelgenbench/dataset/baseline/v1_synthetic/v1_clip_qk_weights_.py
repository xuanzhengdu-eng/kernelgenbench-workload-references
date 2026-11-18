"""Megatron local Q/K weight clipping and attention-logit state reset.

Official source (NVIDIA Megatron-LM, pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/attention.py#L1531-L1578
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/attention.py#L1580-L1626
The tensor-parallel partition's QKV rows are grouped as
``[group, Q + K + V, hidden]``.  If any group exceeds the threshold,
``eta = min(threshold / group_max, 1)``; Q is scaled by ``eta ** alpha`` and K
by ``eta ** (1 - alpha)``, while V is unchanged.  The optional independent
master weight (``linear_qkv.weight.main_param`` in the official method) is
clipped with the same factors.  ``state['eta']``/``state['eta_extended']`` are
updated only when clipping occurs, and ``current_max_attn_logits`` is reset to
``None`` unconditionally.  Mutates weights and state in place and returns
``None``; no collective is performed (the outer qk_clip helper owns that).
"""
import torch


def v1_clip_qk_weights_(weight, main_weight, state, num_groups, q_size, kv_size,
                        threshold, alpha):
    logits = state["current_max_attn_logits"]
    assert logits is not None and logits.ndim == 1 and logits.numel() % num_groups == 0
    assert num_groups > 0 and q_size % num_groups == 0 and kv_size % num_groups == 0
    assert threshold > 0 and 0 <= alpha <= 1
    assert weight.ndim == 2 and weight.shape[0] == q_size + 2 * kv_size
    assert weight.is_contiguous()
    assert logits.device == weight.device and bool((logits >= 0).all())
    if main_weight is not None:
        assert main_weight.shape == weight.shape
        assert main_weight.device == weight.device and main_weight.is_contiguous()

    grouped_max = logits.view(num_groups, -1).max(dim=1).values
    if bool(torch.any(grouped_max > threshold)):
        eta = torch.clamp(threshold / grouped_max, max=1.0).view(num_groups, 1, 1)
        q_per_group = q_size // num_groups
        k_per_group = kv_size // num_groups
        state["eta"] = eta
        state["eta_extended"] = eta.repeat(1, q_per_group, 1)
        with torch.no_grad():
            for tensor in (main_weight, weight):
                if tensor is None:
                    continue
                view = tensor.view(num_groups, (q_size + 2 * kv_size) // num_groups, -1)
                view[:, :q_per_group, :].mul_(state["eta_extended"].pow(alpha))
                view[:, q_per_group:q_per_group + k_per_group, :].mul_(
                    eta.pow(1 - alpha))
    state["current_max_attn_logits"] = None
    return None
