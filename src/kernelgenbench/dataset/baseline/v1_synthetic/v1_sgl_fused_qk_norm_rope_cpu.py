"""SGLang CPU fused per-head Q/K RMSNorm + partial RoPE, in place.

Official source (pinned commit), the exported entry:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/norm.cpp#L1371-L1436
Official source (pinned commit), the per-head norm and both RoPE layouts:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/norm.cpp#L1161-L1299

Contract
--------
* ``q`` is ``[n, num_q_heads * head_dim]`` and ``k`` is
  ``[n, num_kv_heads * head_dim]``; both 2-D, contiguous, same FP16/BF16 dtype.
  There is **no** gate half here and no fused Q/gate split: this is a different
  signature and a different side-effect contract from the vLLM entry.
* Per head: FP32 RMSNorm over the full ``head_dim`` with ``weight`` (no beta),
  rounded back to the input dtype.
* Then partial RoPE over the first ``rotary_dim`` features.  ``is_neox=True``
  rotates the split halves ``[d, d + half)``; ``is_neox=False`` rotates the
  interleaved pairs ``(2d, 2d + 1)``.  cos/sin come from ``cos_sin_cache[pos]``,
  packed as ``[cos | sin]`` with dtype equal to the input.
* In place: the same tensors are updated and ``None`` is returned.
* ``positions`` may be int32 or int64; ``rotary_dim`` must be positive, even and
  ``<= head_dim``.
"""
import torch


def v1_sgl_fused_qk_norm_rope_cpu(q, k, q_weight, k_weight, eps, is_neox,
                                  positions, cos_sin_cache, rotary_dim):
    assert q.ndim == 2 and k.ndim == 2 and q.shape[0] == k.shape[0]
    assert q.dtype == k.dtype == q_weight.dtype == k_weight.dtype == cos_sin_cache.dtype
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.is_contiguous() and k.is_contiguous()
    assert positions.ndim == 1 and positions.shape[0] == q.shape[0]
    assert positions.dtype in (torch.int32, torch.int64)
    assert cos_sin_cache.ndim == 2 and cos_sin_cache.shape[1] == rotary_dim
    head_dim = q_weight.numel()
    assert k_weight.numel() == head_dim and head_dim > 0
    assert q.shape[1] % head_dim == 0 and k.shape[1] % head_dim == 0
    assert 0 < rotary_dim <= head_dim and rotary_dim % 2 == 0

    num_tokens = q.shape[0]
    if num_tokens == 0:
        return None
    half = rotary_dim // 2

    def update(x, weight, heads):
        z = x.reshape(num_tokens, heads, head_dim).float()
        z = (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
             * weight.float()).to(x.dtype).float()
        index = positions.to(torch.long)
        cos = cos_sin_cache[index, :half].float().unsqueeze(1)
        sin = cos_sin_cache[index, half:rotary_dim].float().unsqueeze(1)
        if is_neox:
            left = z[..., :half].clone()
            right = z[..., half:rotary_dim].clone()
            z[..., :half] = left * cos - right * sin
            z[..., half:rotary_dim] = right * cos + left * sin
        else:
            even = z[..., :rotary_dim:2].clone()
            odd = z[..., 1:rotary_dim:2].clone()
            z[..., :rotary_dim:2] = even * cos - odd * sin
            z[..., 1:rotary_dim:2] = odd * cos + even * sin
        x.copy_(z.to(x.dtype).reshape(num_tokens, heads * head_dim))

    update(q, q_weight, q.shape[1] // head_dim)
    update(k, k_weight, k.shape[1] // head_dim)
    return None
