"""SGLang MLA unquantized BF16 fused QKV projection, norm and interleaved RoPE.

Official source (pinned commit), the fused entry ``qkv_proj_with_rope_fused_weight``:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/qkv_proj.cpp#L623-L690
Official source (pinned commit), the body (stage 1 projection, in-place RMSNorm,
q_b projection, absorbed bmm, RoPE):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/qkv_proj.cpp#L416-L620

Contract
--------
* Logical, unpacked BF16 matrices: ``qkv_a_proj_weight`` ``[q_lora + kv_lora +
  qk_rope, hidden]``, ``q_b_proj_weight`` ``[heads * (nope + rope), q_lora]``,
  ``w_kc`` ``[heads, kv_lora, nope]``.  The kernel itself requires VNNI-packed
  weights (``is_vnni=True``); the official loader obtains that layout from
  ``torch.ops.sgl_kernel.convert_weight_packed``, and this reference takes the
  logical layout so the shape relations stay readable.
* ``kv_a_proj_weight`` is the ``[kv_lora + qk_rope, hidden]`` slice of the fused
  weight; its first ``kv_lora`` rows are the compressed KV latent and the rest is
  the shared RoPE key.
* Per row: RMSNorm over ``q_lora`` and over ``kv_lora`` (FP32 statistics, FP32
  weight multiply, rounded back to BF16), then ``q_b`` projection, then the
  absorbed ``w_kc`` contraction on the nope part.
* Interleaved RoPE over the trailing ``rope`` features for Q per head and for K;
  ``rope % 32 == 0`` is required by the kernel.
* Returns ``(q, k, v)``; ``v`` is a storage-sharing view of the first ``kv_lora``
  columns of ``k`` (the C++ returns ``k_input.narrow(-1, 0, kv_lora)``).
"""
import torch


def v1_sgl_mla_qkv_rope(hidden_states, qkv_a_proj_weight, q_b_proj_weight,
                        w_kc, q_a_layernorm_weight, kv_a_layernorm_weight,
                        positions, cos_sin_cache, eps):
    x = hidden_states
    heads = w_kc.shape[0]
    q_lora = q_a_layernorm_weight.numel()
    kv_lora = w_kc.shape[1]
    nope = w_kc.shape[2]
    rope = cos_sin_cache.shape[1]
    assert x.dtype == torch.bfloat16 and x.ndim == 2
    assert rope % 32 == 0
    assert qkv_a_proj_weight.shape == (q_lora + kv_lora + rope, x.shape[1])
    assert q_b_proj_weight.shape == (heads * (nope + rope), q_lora)
    assert positions.dtype == torch.int64 and positions.shape == (x.shape[0],)

    def linear(a, weight):
        # The packed kernel multiplies by the transposed weight in BF16 storage.
        return (a.float() @ weight.float().T).to(x.dtype)

    def rms_norm(z, weight):
        f = z.float()
        return (f * torch.rsqrt((f * f).mean(-1, keepdim=True) + eps)
                * weight.float()).to(x.dtype)

    projected = linear(x, qkv_a_proj_weight)
    qa = rms_norm(projected[:, :q_lora], q_a_layernorm_weight)
    kv_latent = rms_norm(projected[:, q_lora:q_lora + kv_lora], kv_a_layernorm_weight)
    k_rope_source = projected[:, q_lora + kv_lora:]

    qb = linear(qa, q_b_proj_weight).reshape(x.shape[0], heads, nope + rope)
    q_nope = torch.einsum('bhd,hkd->bhk', qb[:, :, :nope].float(), w_kc.float()).to(x.dtype)

    table = cos_sin_cache[positions].float()
    cos = table[:, :rope // 2]
    sin = table[:, rope // 2:]

    def rotate(z, per_head):
        z = z.float()
        c = cos[:, None, :] if per_head else cos
        s = sin[:, None, :] if per_head else sin
        out = torch.empty_like(z)
        out[..., 0::2] = z[..., 0::2] * c - z[..., 1::2] * s
        out[..., 1::2] = z[..., 1::2] * c + z[..., 0::2] * s
        return out.to(x.dtype)

    q = torch.cat((q_nope, rotate(qb[:, :, nope:], True)), dim=-1)
    k = torch.cat((kv_latent, rotate(k_rope_source, False)), dim=-1).unsqueeze(1)
    return q, k, k[..., :kv_lora]
