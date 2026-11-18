"""vLLM fused Q/K RMSNorm, partial RoPE and raw gate split (Qwen-style attention).

Official source (pinned commit), fused entry and its argument checks:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_qk_norm_rope.py#L139-L280
Official source (pinned commit), the Triton kernel that defines the arithmetic:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_qk_norm_rope.py#L17-L136

Contract
--------
* ``q_gate`` is ``[n, num_q_heads * 2 * head_dim]`` laid out per head as
  ``[q | gate]``; ``k`` is ``[n, num_kv_heads * head_dim]``.
* Per head: RMSNorm over the full ``head_dim`` using ``weight + norm_beta``,
  rounded back to the input dtype before the rotation (the kernel round-trips
  through INPUT_DTYPE to match the unfused norm -> memory -> rope path).
* Partial NeoX RoPE over the first ``rotary_dim`` features;
  ``[rotary_dim, head_dim)`` is RMSNorm-only pass-through.
* ``positions`` is ``[n]`` or ``[3, n]`` (MRoPE); with MRoPE the occurrence of
  dimension ``i`` in the interleaved T/H/W pattern picks the T, H or W row.
* Returns three freshly allocated contiguous tensors
  ``(q_out, k_out, gate_out)``; ``gate_out`` is the raw (pre-sigmoid) gate copy.
* ``n == 0`` returns empty tensors.  Invalid rotary dims, dtypes, strides and
  MRoPE arguments raise ``ValueError`` (upstream's own checks, not benchmark
  checks).

The separate SGLang CPU in-place branch of the old fork reference (``backend=
'sgl_cpu'``, different contract: writes into q/k, returns ``None``, no gate) is
recorded in decision.json as a blocked branch: the pinned SGLang container has no
CPU ``sgl_kernel`` extension (``torch.ops.sgl_kernel`` exposes no
``fused_qk_norm_rope_cpu``), so it cannot be executed here.
"""
import torch


def v1_vllm_qk_rmsnorm_rope_gate(
    q_gate,
    k,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    eps,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    mrope_section=None,
    norm_beta=0.0,
):
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            f"rotary_dim must be a positive even integer <= head_dim, got "
            f"rotary_dim={rotary_dim}, head_dim={head_dim}")
    if q_gate.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q_gate.dtype:
        raise ValueError(
            f"q_gate and k must have the same FP16 or BF16 dtype, got "
            f"{q_gate.dtype} and {k.dtype}")
    for name, tensor in (("q_gate", q_gate), ("k", k), ("q_weight", q_weight),
                         ("k_weight", k_weight), ("cos_sin_cache", cos_sin_cache)):
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous in its last dimension")

    n = q_gate.shape[0]
    if positions.ndim not in (1, 2):
        raise ValueError(f"positions must be 1D or 2D, got shape={positions.shape}")
    if positions.shape[-1] != n:
        raise ValueError(
            f"positions token dimension must match q_gate, got {positions.shape[-1]} and {n}")
    has_mrope = positions.ndim == 2
    if has_mrope:
        if positions.shape[0] != 3:
            raise ValueError(f"MRoPE positions must have shape (3, n_tokens), got {positions.shape}")
        if mrope_section is None or len(mrope_section) != 3:
            raise ValueError("mrope_section must contain the T/H/W frequency counts")
        if sum(mrope_section) != rotary_dim // 2:
            raise ValueError(
                "mrope_section must sum to rotary_dim // 2, got "
                f"{mrope_section} and rotary_dim={rotary_dim}")
        section_h, section_w = mrope_section[1], mrope_section[2]
    else:
        if mrope_section is not None:
            raise ValueError("mrope_section requires 2D MRoPE positions")
        section_h = section_w = 0

    dtype = q_gate.dtype
    q_out = torch.empty((n, num_q_heads * head_dim), dtype=dtype, device=q_gate.device)
    k_out = torch.empty((n, num_kv_heads * head_dim), dtype=dtype, device=k.device)
    gate_out = torch.empty_like(q_out)
    if n == 0:
        return q_out, k_out, gate_out

    half = rotary_dim // 2
    q_heads = q_gate.reshape(n, num_q_heads, 2, head_dim)
    q = q_heads[:, :, 0]
    gate = q_heads[:, :, 1].reshape(n, num_q_heads * head_dim).contiguous()
    key = k.reshape(n, num_kv_heads, head_dim)
    offsets = torch.arange(half, device=q_gate.device)

    def transform(x, weight):
        xf = x.float()
        norm = ((xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
                 * (weight.float() + norm_beta)).to(dtype).float())
        if has_mrope:
            pos = positions[0].to(torch.long)[:, None].expand(n, half)
            is_h = (offsets % 3 == 1) & (offsets < 3 * section_h)
            is_w = (offsets % 3 == 2) & (offsets < 3 * section_w)
            pos = torch.where(is_h[None, :], positions[1].to(torch.long)[:, None], pos)
            pos = torch.where(is_w[None, :], positions[2].to(torch.long)[:, None], pos)
        else:
            pos = positions.to(torch.long)[:, None].expand(n, half)
        columns = offsets[None, :].expand(n, half)
        cos = cos_sin_cache[pos, columns].float().unsqueeze(1)
        sin = cos_sin_cache[pos, columns + half].float().unsqueeze(1)
        left = norm[..., :half]
        right = norm[..., half:rotary_dim]
        rotated = torch.cat((left * cos - right * sin, right * cos + left * sin,
                             norm[..., rotary_dim:]), -1)
        return rotated.to(dtype).reshape(n, -1).contiguous()

    q_out = transform(q, q_weight)
    k_out = transform(key, k_weight)
    return q_out, k_out, gate
