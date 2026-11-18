"""Megatron multimodal RoPE embedding and context-parallel partition.

Official NVIDIA entry (the old reference cited the flagos-ai fork):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/rotary_pos_embedding.py#L314-L367

Contract, as upstream: ``position_ids`` is [3, batch, seq] T/H/W; ``inv_freq``
has one entry per rotary pair; ``mrope_section`` (three ints) sums to the
number of rotary pairs and is doubled internally before splitting the
2*rotary_dim buffer, so each axis owns a contiguous block of frequencies.
Output is [seq, batch, 1, 2*rotary_dim] (sequence-major, as Megatron's BSHD
consumers expect).  With ``cp_size > 1`` the sequence is cut in two halves and
rank ``cp_rank`` keeps the same-order front half plus the reversed-order back
half, exactly like ``get_pos_emb_on_this_cp_rank``.

The chunk order below (``mrope_section * 2`` repeats the three widths, so the
split is T,H,W,T,H,W) is upstream's, and matches the old reference for its
default branch bit for bit.  Fork-only behaviour of the old reference is not
reproduced: the ``interleaved_mrope=True`` axis switch that no NVIDIA entry has,
and the ``interpolation`` argument, which upstream names
``seq_len_interpolation_factor``.
"""
import torch


def v1_megatron_mrope_freqs(position_ids, inv_freq, mrope_section,
                            rotary_interleaved=False,
                            seq_len_interpolation_factor=None,
                            cp_size=1, cp_rank=0):
    seq = position_ids.to(device=inv_freq.device, dtype=inv_freq.dtype)
    if seq_len_interpolation_factor is not None:
        # Upstream scales the converted tensor in place.  ``.to`` returns the
        # caller's tensor when dtype and device already match, so a float32
        # position_ids is updated in place; an integer one is copied first.
        seq *= 1 / seq_len_interpolation_factor
    inv_freq_expanded = inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
    seq_expanded = seq[:, :, None, :].float()
    freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)
    if not rotary_interleaved:
        emb = torch.cat((freqs, freqs), dim=-1)
    else:
        bs = freqs.shape[1]
        emb = torch.stack(
            (freqs.view(3, bs, -1, 1), freqs.view(3, bs, -1, 1)), dim=-1
        ).view(3, bs, freqs.shape[0], -1)
    sections = list(mrope_section) * 2
    emb = torch.cat(
        [chunk[i % 3] for i, chunk in enumerate(emb.split(sections, dim=-1))], dim=-1
    )
    emb = emb[..., None, :].transpose(0, 1).contiguous()
    if cp_size > 1:
        emb = emb.view(2 * cp_size, -1, *emb.shape[1:])
        idx = torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], device=emb.device)
        emb = emb.index_select(0, idx).view(-1, *emb.shape[2:])
    return emb
