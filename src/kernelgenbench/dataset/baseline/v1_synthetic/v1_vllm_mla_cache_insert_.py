"""vLLM MLA latent+RoPE concatenation into the paged KV cache (CUDA).

Official implementation, vLLM commit
f358f0578a83f4e1fe6fc49a492d3ae9362de8e5:

* Python launcher (``vllm._custom_ops.concat_and_cache_mla``):
  https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L3252-L3270
* CUDA launcher:
  https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/cache_kernels.cu#L1050-L1150
* CUDA kernel:
  https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/cache_kernels.cu#L490-L536

Contract: ``concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping,
kv_cache_dtype, scale, skip_scale=False)`` writes the concatenated row
``[kv_c[t] | k_pe[t]]`` into the paged cache row ``slot_mapping[t]``, addressed
as ``kv_cache[slot // block_size, slot % block_size]``.  Negative slots are
padding and are skipped; the function returns ``None`` and mutates
``kv_cache`` in place.

* ``kv_c`` and ``k_pe`` are 2-D with unit last-dim stride.  ``stride(0)`` may
  exceed the row width and only the first ``slot_mapping.size(0)`` rows are
  read (vLLM V1 pads them for CUDA graphs), see the comment at
  csrc/cache_kernels.cu#L1058-L1067.
* ``kv_cache`` is 3-D with ``size(-1) == kv_c.size(1) + k_pe.size(1)`` (the
  launcher enforces this at csrc/cache_kernels.cu#L1104-L1105) and unit
  last-dim stride; block and entry strides are taken from the tensor, so a
  row-padded backing storage is supported.
* ``kv_cache_dtype="auto"`` stores the input dtype verbatim (float32,
  float16 or bfloat16, see DISPATCH_BY_KV_CACHE_DTYPE).  ``"fp8"`` stores
  E4M3 bytes in a uint8 cache after dividing by the scalar float32 ``scale``;
  ``skip_scale=True`` makes the divisor 1.0.  The conversion is
  ``__NV_SATFINITE`` E4M3 (NaN -> NaN, +/-Inf -> +/-448), matching
  ``Tensor.to(torch.float8_e4m3fn)`` for every value.

Only the CUDA backend is part of this contract.  The pinned CUDA build has no
CPU dispatch for this op: its CPU counterpart is the *separate*
``_C.concat_and_cache_mla_cpu`` op behind
``vllm._custom_ops.amx_mla_concat_and_cache`` (csrc/cpu/torch_bindings.cpp
#L135-L139 and #L795-L798, csrc/cpu/sgl-kernels/mla_cache.cpp#L60-L111) with a
different contract (no dtype/scale, int32/int64 slots, rank-3 ``k_pe``, one
input row per token) and it is not registered in this build.
``concat_and_cache_mla_grouped`` is a separate entry point as well
(csrc/cache_kernels.cu#L1155-L1258) taking an int64 device pointer array plus
per-layer fp8 scales; it is not part of this single-cache contract.
"""
import torch


def v1_vllm_mla_cache_insert_(kv_c, k_pe, kv_cache, slot_mapping,
                              kv_cache_dtype="auto", scale=None,
                              skip_scale=False):
    assert kv_c.device.type == "cuda", "concat_and_cache_mla is CUDA-only"
    assert kv_c.ndim == 2 and k_pe.ndim == 2 and kv_cache.ndim == 3
    assert kv_c.dtype == k_pe.dtype
    assert kv_c.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert kv_c.stride(-1) == k_pe.stride(-1) == 1
    assert kv_cache.stride(-1) == 1
    assert kv_cache.device == kv_c.device == k_pe.device
    assert slot_mapping.ndim == 1 and slot_mapping.dtype == torch.int64
    assert slot_mapping.device == kv_c.device

    # The launcher uses slot_mapping.size(0) as the token count.
    num_tokens = slot_mapping.shape[0]
    assert kv_c.shape[0] >= num_tokens and k_pe.shape[0] >= num_tokens
    rank, pe_dim = kv_c.shape[1], k_pe.shape[1]
    assert kv_cache.shape[-1] == rank + pe_dim
    block_size = kv_cache.shape[1]
    assert block_size > 0 and kv_cache.shape[0] > 0

    if kv_cache_dtype in ("auto", "float16", "bfloat16"):
        # These spellings all select Fp8KVCacheDataType::kAuto.
        assert kv_cache.dtype == kv_c.dtype
        divisor = None
    elif kv_cache_dtype == "fp8":
        assert kv_cache.dtype == torch.uint8
        assert scale is not None and scale.dtype == torch.float32
        assert scale.numel() == 1 and scale.device == kv_c.device
        divisor = 1.0 if skip_scale else float(scale.item())
    else:
        raise ValueError(f"unsupported kv_cache_dtype {kv_cache_dtype!r}")

    for token in range(num_tokens):
        slot = int(slot_mapping[token].item())
        if slot < 0:
            continue
        assert 0 <= slot < kv_cache.shape[0] * block_size
        row = kv_cache[slot // block_size, slot % block_size]
        if divisor is None:
            row[:rank].copy_(kv_c[token])
            row[rank:].copy_(k_pe[token])
        else:
            value = torch.cat((kv_c[token], k_pe[token])).float() / divisor
            row.copy_(value.clamp(-448.0, 448.0)
                      .to(torch.float8_e4m3fn).view(torch.uint8))
    return None
