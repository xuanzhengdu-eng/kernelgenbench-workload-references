"""Reference: scatter KV locations into per-request cache ranges (Triton backend).

Official implementation (pinned SGLang commit
``62384de1ae5e3bbd3159afd1639525b0bf39fbf9``):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/mem_cache/allocation.py#L733-L765

This is the CUDA backend of the same logical entry as
``v1_scatter_request_cache_ranges_`` (the CPU backend).  The dispatcher
``assign_req_to_token_pool_func`` (same file, L779-L797) launches this
``@triton.jit`` kernel with ``grid=(batch_size,)`` and
``pool_len``/``bs_upper`` as ``tl.constexpr`` whenever the batch is not on CPU.

Contract, as the pinned Triton kernel states it:

* ``req_pool_indices``/``start_offset``/``end_offset`` are 1-D index tensors on
  the GPU; ``req_to_token`` is a 2-D int32 table updated in place and
  ``out_cache_loc`` is read only; the function returns ``None``;
* ``pool_len`` must be the table's second dimension, and ``bs_upper`` (a
  constexpr) must be at least the batch size; the kernel reads the offsets of
  all earlier requests to build the exclusive prefix sum, masked by
  ``length_offset < pid``;
* for request ``pid``, slot ``j`` in ``[start, end)`` becomes
  ``req_to_token[req, j] = int32(out_cache_loc[prefix + j - start])``;
* slots outside the ranges keep their previous value.

Unlike the CPU backend this kernel performs no dtype/dimension validation (the
caller owns it) and truncates the int64 cache locations to int32 on store; the
observable result for in-range values is identical.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _assign_req_to_token_pool_kernel(req_pool_indices, req_to_token, start_offset,
                                     end_offset, out_cache_loc,
                                     pool_len: tl.constexpr, bs_upper: tl.constexpr):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


def v1_scatter_request_cache_ranges_triton(req_pool_indices, req_to_token,
                                           start_offset, end_offset, out_cache_loc):
    assert req_to_token.dtype == torch.int32 and req_to_token.ndim == 2
    assert out_cache_loc.dtype == torch.int64
    assert req_pool_indices.dtype in (torch.int32, torch.int64)
    assert start_offset.dtype in (torch.int32, torch.int64)
    assert end_offset.dtype == start_offset.dtype
    assert all(t.device.type == "cuda" and t.is_contiguous()
               for t in (req_pool_indices, req_to_token, start_offset, end_offset,
                         out_cache_loc))
    batch = req_pool_indices.numel()
    assert start_offset.numel() == end_offset.numel() == batch
    required = int((end_offset.to(torch.int64) - start_offset.to(torch.int64)).sum())
    assert required <= out_cache_loc.numel()
    if batch:
        _assign_req_to_token_pool_kernel[(batch,)](
            req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc,
            req_to_token.shape[1], triton.next_power_of_2(batch))
    return None
