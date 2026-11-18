"""Megatron BSHD native RoPE (official NVIDIA entry).

https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/rope_utils.py#L92-L129
``freqs`` holds real angular frequencies and the last dimension selects the
partial rotary width; cosine and sine are taken inside the function.  The
interleaved pairing follows upstream, which indexes exactly four dimensions.
The old reference's ``inverse`` flag has no NVIDIA counterpart and is removed.
"""
import torch


def _rotate_half(x, rotary_interleaved):
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    assert x.dim() == 4, "Megatron's interleaved pairing indexes [S, B, H, D]"
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def v1_rope_bshd(t, freqs, rotary_interleaved=False, multi_latent_attention=False,
                 mscale=1.0):
    rot_dim = freqs.shape[-1]
    x, tail = t[..., :rot_dim], t[..., rot_dim:]
    if multi_latent_attention:
        x = torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)
    cos_ = (torch.cos(freqs) * mscale).to(x.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(x.dtype)
    y = (x * cos_) + (_rotate_half(x, rotary_interleaved) * sin_)
    return torch.cat((y, tail), dim=-1)
