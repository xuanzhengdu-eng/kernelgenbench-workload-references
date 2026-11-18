"""Compact accepted KV locations into a caller-owned output buffer (CUDA/Triton branch).

https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/spec.cpp#L528-L554
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/speculative/eagle.py#L55-L89

The pinned entry ``fill_accept_out_cache_loc_func`` dispatches on the tensor
device: CPU tensors use the C++ ``fill_accept_out_cache_loc_cpu`` kernel, CUDA
tensors use the Triton ``fill_accept_out_cache_loc`` kernel.  This is the same
logical entry as ``v1_accept_cache_locations_`` with the same gather/skip/-1
semantics, so both references share one target Item.  The CUDA branch requires
CUDA, contiguous int32/int64 indices and int64 values/output; the output tail
beyond the number of accepted entries is left untouched.  Returns None.
"""
import torch


def v1_accept_cache_locations_triton(accept_index, out_cache_loc, accept_out_cache_loc):
    assert accept_index.device.type == "cuda"
    assert out_cache_loc.device.type == accept_out_cache_loc.device.type == "cuda"
    assert accept_index.dtype in (torch.int32, torch.int64)
    assert out_cache_loc.dtype == accept_out_cache_loc.dtype == torch.int64
    assert all(x.is_contiguous() for x in (accept_index, out_cache_loc, accept_out_cache_loc))
    assert accept_out_cache_loc.numel() >= accept_index.numel()
    indices = accept_index.reshape(-1)
    valid = indices[indices >= 0].to(torch.long)
    if valid.numel() and int(valid.max()) >= out_cache_loc.numel():
        raise IndexError("accept index out of range")
    accept_out_cache_loc.view(-1)[:valid.numel()].copy_(out_cache_loc.view(-1)[valid])
    return None
