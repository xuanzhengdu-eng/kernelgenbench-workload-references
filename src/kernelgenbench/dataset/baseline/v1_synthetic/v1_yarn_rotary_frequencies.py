"""Megatron YaRN rotary frequency table and concentration factor (CP size one).

Official source (pinned commit), the table itself:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py#L106-L158
Official source (pinned commit), correction range and concentration factor:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py#L218-L272

``YarnRotaryEmbedding.get_emb`` returns ``(emb, mscale)`` where ``emb`` is FP32
``[seq_len, 1, 1, dim]`` with the frequency halves repeated and ``mscale`` is a
Python float.  This contract covers ``rotary_percent=1.0``, the non-interleaved
layout (the upstream class asserts it) and CP group size one.  The returned table
is a *frequency* table: no sine/cosine is applied here.  ``device`` only selects
where the module buffers start; upstream ``get_emb`` migrates ``inv_freq`` to the
current CUDA device on first use, so the table is produced there.
"""
import math

import torch


def v1_yarn_rotary_frequencies(
    seq_len,
    offset,
    dim,
    base,
    scaling,
    original_max,
    beta_fast,
    beta_slow,
    mscale,
    mscale_all_dim,
    round_range,
    device,
):
    assert seq_len >= 0 and dim > 0 and dim % 2 == 0 and dim % 4 == 0
    assert base > 1 and scaling > 0 and original_max > 0
    assert beta_fast > 0 and beta_slow > 0

    def correction_dim(rotations):
        return dim * math.log(original_max / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = correction_dim(beta_fast)
    high = correction_dim(beta_slow)
    if round_range:
        low = math.floor(low)
        high = math.ceil(high)
    low = max(low, 0)
    high = min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float32, device=device) - low)
            / (high - low)).clamp(0, 1)
    mask = 1.0 - ramp

    exponent = torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
    inv_freq_extra = 1.0 / (base ** exponent)
    inv_freq_inter = 1.0 / (scaling * base ** exponent)
    inv_freq = inv_freq_inter * (1 - mask) + inv_freq_extra * mask

    seq = torch.arange(seq_len, dtype=torch.float32, device=device) + offset
    freqs = torch.outer(seq, inv_freq)

    def get_mscale(scale, factor):
        if scale <= 1:
            return 1.0
        return 0.1 * factor * math.log(scale) + 1.0

    if mscale is None or mscale_all_dim is None:
        factor = get_mscale(scaling, 1.0)
    else:
        factor = get_mscale(scaling, mscale) / get_mscale(scaling, mscale_all_dim)

    emb = torch.cat((freqs, freqs), dim=-1)
    return emb[:, None, None, :], float(factor)
