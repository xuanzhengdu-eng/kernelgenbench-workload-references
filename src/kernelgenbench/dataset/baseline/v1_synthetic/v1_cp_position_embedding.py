"""Megatron context-parallel positional-embedding selection (seq_dim=1).

Official source (NVIDIA/Megatron-LM pinned commit ``266f1c97``):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/rope_utils.py#L48-L70

Contract
--------
``pos_emb`` is ``[B, T, H]`` with ``seq_dim=1``; the sequence must be divisible
by ``2 * cp_size``.  The function views the sequence axis as
``[2*cp_size, T/(2*cp_size)]``, selects the two chunks ``cp_rank`` and
``2*cp_size - cp_rank - 1`` (a mirrored pair) and concatenates them along the
sequence axis, returning a tensor of shape ``[B, 2*T/(2*cp_size), H]``.  The
process group is represented by the scalar ``cp_size``/``cp_rank`` because this
selection performs no communication; upstream obtains both from the CP process
group.  The result is a fresh contiguous tensor built from the two chunks.
"""
import torch


def v1_cp_position_embedding(pos_emb, cp_size, cp_rank):
    n = pos_emb.shape[1] // (2 * cp_size)
    left = pos_emb[:, cp_rank * n:(cp_rank + 1) * n]
    mirrored = 2 * cp_size - cp_rank - 1
    right = pos_emb[:, mirrored * n:(mirrored + 1) * n]
    return torch.cat((left, right), dim=1)
