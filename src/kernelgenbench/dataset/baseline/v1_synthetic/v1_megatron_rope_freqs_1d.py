"""Megatron RotaryEmbedding: 1D position-frequency table and optional CP split.

Source (official fixed commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/rotary_pos_embedding.py#L150-L206
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/rope_utils.py#L48-L70

``inv_freq`` is the precomputed float32 buffer already resident on the input
device; initialization, Llama scaling, CPU-to-GPU migration and lru_cache are
out of scope.  ``cp_size``/``cp_rank`` describe an already-created CP process
group; no collectives occur.  The function is the functional sub-contract of
``RotaryEmbedding.get_emb`` plus ``RotaryEmbedding.forward``'s CP slice.
"""
import torch


def v1_megatron_rope_freqs_1d(inv_freq, max_seq_len, offset=0, interpolation=None,
                             interleaved=False, packed_seq=False, cp_size=1, cp_rank=0):
    seq = torch.arange(max_seq_len, device=inv_freq.device, dtype=inv_freq.dtype) + offset
    if interpolation is not None:
        seq = seq * (1 / interpolation)
    freq = torch.outer(seq, inv_freq)
    if interleaved:
        out = torch.stack((freq, freq), dim=-1).reshape(max_seq_len, -1)
    else:
        out = torch.cat((freq, freq), dim=-1)
    out = out[:, None, None, :]
    if cp_size > 1 and not packed_seq:
        out = out.view(2 * cp_size, -1, 1, 1, out.shape[-1])
        index = torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], device=out.device)
        out = out.index_select(0, index).reshape(-1, 1, 1, out.shape[-1])
    return out
