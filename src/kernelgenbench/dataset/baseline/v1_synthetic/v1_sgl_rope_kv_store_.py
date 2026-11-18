# SGLang pinned apply_rope_inplace_with_kvcache: in-place full-head RoPE on q
# and k (NeoX or interleaved layout), then store rotated k and unmodified v in
# flat K/V cache rows selected by out_loc.  Unique valid slots only.
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/attention/rope.py#L174-L209
import torch


def v1_sgl_rope_kv_store_(q, k, v, k_cache, v_cache, cos_sin_cache, positions,
                          out_loc, is_neox, rope_dim=0):
    """In-place SGLang full-head RoPE and flat K/V cache scatter (unique valid slots)."""
    d = rope_dim or cos_sin_cache.shape[-1]
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == d and d % 2 == 0
    assert q.element_size() == k.element_size()
    assert k.dtype == v.dtype == k_cache.dtype == v_cache.dtype
    c, s = cos_sin_cache[positions.long()].float().chunk(2, -1)

    def rotate(x):
        if is_neox:
            u, w = x.float().chunk(2, -1)
            return torch.cat((u * c[:, None, :] - w * s[:, None, :],
                              u * s[:, None, :] + w * c[:, None, :]), -1).to(x.dtype)
        u = x[..., 0::2].float()
        w = x[..., 1::2].float()
        return torch.stack((u * c[:, None, :] - w * s[:, None, :],
                            u * s[:, None, :] + w * c[:, None, :]), -1).flatten(-2).to(x.dtype)

    q.copy_(rotate(q))
    k.copy_(rotate(k))
    slots = out_loc.long()
    k_cache[slots] = k.flatten(1)
    v_cache[slots] = v.flatten(1)
    return None
