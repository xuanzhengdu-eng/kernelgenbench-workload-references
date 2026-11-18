"""Native DeepSeek scaling rotary embedding. Returns fresh Q and K tensors."""
# Source: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/rotary_embedding/rope_variant.py#L472-L505
import torch


def v1_sgl_deepseek_scaling_rope_native(rotary, positions, query, key, offsets=None):
    head_size, rotary_dim = rotary.head_size, rotary.rotary_dim
    cos_sin_cache, is_neox_style = rotary.cos_sin_cache, rotary.is_neox_style
    index = positions if offsets is None else positions + offsets
    cosine, sine = cos_sin_cache[index].chunk(2, dim=-1)
    if is_neox_style:
        cosine = cosine.repeat(1, 1, 2).unsqueeze(-2)
        sine = sine.repeat(1, 1, 2).unsqueeze(-2)
    else:
        cosine = cosine.repeat_interleave(2, dim=-1).unsqueeze(-2)
        sine = sine.repeat_interleave(2, dim=-1).unsqueeze(-2)

    def apply(x):
        prefix = x[..., :rotary_dim]
        if is_neox_style:
            left, right = prefix.chunk(2, dim=-1)
            rotated = torch.cat((-right, left), dim=-1)
        else:
            rotated = torch.stack((-prefix[..., 1::2], prefix[..., ::2]), dim=-1).flatten(-2)
        out = prefix * cosine + rotated * sine
        if rotary_dim < head_size:
            out = torch.cat((out, x[..., rotary_dim:]), dim=-1)
        return out.to(query.dtype)

    return apply(query), apply(key)
