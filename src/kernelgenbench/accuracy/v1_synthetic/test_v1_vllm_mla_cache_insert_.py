"""Accuracy and performance for v1_vllm_mla_cache_insert_.

Covers the two branches the pinned CUDA build really executes for
``vllm._custom_ops.concat_and_cache_mla``:

* ``kv_cache_dtype="auto"``: verbatim row copy for float32/float16/bfloat16.
* ``kv_cache_dtype="fp8"``: E4M3 bytes written into a uint8 cache after a
  float32 division by ``scale`` (``skip_scale=True`` divides by 1.0).

Every case uses a paged cache whose backing storage is row-padded, permuted
and negative slot mappings, and more input rows than slots (vLLM V1 CUDA-graph
padding).  The test checks the exact cache bytes, that padded rows/columns and
untouched rows survive, that the cache object and storage identity are
unchanged, and that the inputs are not modified.  The removed CPU/grouped
branches are documented in the reference module docstring.
"""
import statistics
import time

import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# tokens, kv_lora_rank, pe_dim, input dtype, cache format, fp8 scale,
# skip_scale, block_size, absolute slot mapping, kv_c/k_pe rows, cache row pad
COMBOS = [
    (5, 512, 64, torch.float32, "auto", 1.0, False, 4, [0, 3, -1, 6, 2], 5, 0),
    (5, 512, 64, torch.float16, "auto", 1.0, False, 4, [9, 1, -1, 4, 0], 7, 7),
    (4, 128, 64, torch.bfloat16, "auto", 1.0, False, 4, [-1, 5, 2, 7], 4, 11),
    (7, 48, 32, torch.float16, "auto", 1.0, False, 4, [11, 0, 5, -1, 9, 3, 7], 8, 0),
    (6, 256, 64, torch.float16, "fp8", 0.5, False, 4, [3, 0, -1, 6, 9, 1], 6, 3),
    (6, 256, 64, torch.bfloat16, "fp8", 2.0, False, 4, [7, 2, 5, -1, 0, 10], 6, 5),
    (4, 512, 64, torch.float32, "fp8", 1.0, False, 4, [4, 1, 3, -1], 4, 0),
    (5, 128, 64, torch.bfloat16, "fp8", 0.25, True, 4, [2, 6, -1, 0, 9], 5, 7),
]


def get_input(case, device=None):
    device = kernelgenbench.device if device is None else device
    (n, rank, pe_dim, dtype, fmt, scale_value, skip_scale, block_size, slots,
     rows, row_pad) = case
    assert fmt in ("auto", "fp8")
    gen = torch.Generator(device="cpu").manual_seed(
        7000 + 31 * n + rank + 3 * pe_dim + row_pad)
    # stride(0) > row width: the kernel must honour kv_c.stride(0)/k_pe.stride(0).
    kv_c = (torch.randn(rows, rank + 3, generator=gen, dtype=torch.float32) * 60.0
            ).to(device=device, dtype=dtype)[:, :rank]
    k_pe = (torch.randn(rows, pe_dim + 5, generator=gen, dtype=torch.float32) * 60.0
            ).to(device=device, dtype=dtype)[:, :pe_dim]
    if fmt == "fp8" and n:
        # Exercise E4M3 saturation (+/-600 -> +/-448) and the scale division.
        kv_c[0, 0] = 600.0
        k_pe[0, 0] = -600.0

    width = rank + pe_dim
    cache_dtype = torch.uint8 if fmt == "fp8" else dtype
    fill = 173 if fmt == "fp8" else 17
    num_blocks = max(slots) // block_size + 2
    storage = torch.full((num_blocks, block_size, width + row_pad), fill,
                         dtype=cache_dtype, device=device)
    return dict(kv_c=kv_c, k_pe=k_pe, kv_cache=storage[..., :width],
                slot_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
                kv_cache_dtype=fmt,
                scale=torch.tensor([scale_value], dtype=torch.float32, device=device),
                skip_scale=skip_scale)


def input_build(config):
    return get_input(config)


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_vllm_mla_cache_insert_(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.triton.v1_vllm_mla_cache_insert_(**kwargs)


def cast_fp8(value):
    """The kernel's __NV_SATFINITE E4M3 conversion, expressed in torch."""
    return value.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def expected_cache(data):
    """Independent expected cache: same slots, same concat, same fp8 bytes."""
    expected = data["kv_cache"].clone()
    rank = data["kv_c"].shape[1]
    block_size = expected.shape[1]
    divisor = 1.0 if data["skip_scale"] else float(data["scale"].item())
    for token, slot in enumerate(data["slot_mapping"].tolist()):
        if slot < 0:
            continue
        row = expected[slot // block_size, slot % block_size]
        if data["kv_cache_dtype"] == "auto":
            row[:rank].copy_(data["kv_c"][token])
            row[rank:].copy_(data["k_pe"][token])
        else:
            value = torch.cat((data["kv_c"][token], data["k_pe"][token])).float() / divisor
            row.copy_(cast_fp8(value).view(torch.uint8))
    return expected


def measure(fn, inputs):
    """Median single-call ms; input copies and warmup stay out of the timed region."""
    samples = []
    for _ in range(6):
        arguments = {name: value.to(device=kernelgenbench.device, copy=True)
                     if isinstance(value, torch.Tensor) else value
                     for name, value in inputs.items()}
        if kernelgenbench.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(**arguments)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            fn(**arguments)
            samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples[1:])


def compute_ref_performance(**kwargs):
    return measure(compute_ref, kwargs)


def compute_res_performance(**kwargs):
    return measure(compute_res, kwargs)


def compare(ref_cache, res_cache, case):
    """Exact contract comparison of the two written caches."""
    fmt = case[4]
    assert ref_cache.shape == res_cache.shape
    assert ref_cache.dtype == res_cache.dtype
    if fmt == "fp8":
        # Byte equality of the E4M3 payload (uint8 cache).
        assert ref_cache.dtype == torch.uint8
        assert torch.equal(ref_cache.contiguous().view(torch.uint8),
                           res_cache.contiguous().view(torch.uint8))
    else:
        assert torch.equal(ref_cache, res_cache)


@label("v1_vllm_mla_cache_insert_")
@parametrize("case", COMBOS)
def test_accuracy_v1_vllm_mla_cache_insert_(case):
    (n, rank, pe_dim, dtype, fmt, _, _, block_size, slots, rows,
     row_pad) = case
    assert kernelgenbench.device.type == "cuda"
    a = input_build(case)
    b = input_build(case)
    width = rank + pe_dim
    for side in (a, b):
        assert side["kv_c"].is_cuda and side["kv_c"].shape == (rows, rank)
        assert side["kv_c"].stride(0) == rank + 3 and side["kv_c"].stride(1) == 1
        assert side["k_pe"].shape == (rows, pe_dim) and side["k_pe"].stride(0) == pe_dim + 5
        assert side["kv_cache"].is_cuda and side["kv_cache"].shape[-1] == width
        assert side["kv_cache"].stride(1) == width + row_pad
        assert side["kv_cache"].dtype == (torch.uint8 if fmt == "fp8" else dtype)
        assert side["slot_mapping"].dtype == torch.int64
        assert side["slot_mapping"].numel() == n

    expected = expected_cache(a)
    storage_a, storage_b = a["kv_cache"]._base, b["kv_cache"]._base
    ptr_a, ptr_b = a["kv_cache"].data_ptr(), b["kv_cache"].data_ptr()
    off_a, off_b = a["kv_cache"].storage_offset(), b["kv_cache"].storage_offset()
    pad_a = storage_a[..., width:].clone()
    pad_b = storage_b[..., width:].clone()
    snap_a = {name: value.clone() for name, value in a.items()
              if isinstance(value, torch.Tensor) and name != "kv_cache"}
    snap_b = {name: value.clone() for name, value in b.items()
              if isinstance(value, torch.Tensor) and name != "kv_cache"}

    ref_out = compute_ref(**a)
    res_out = compute_res(**b)
    assert ref_out is None and res_out is None
    compare(a["kv_cache"], b["kv_cache"], case)

    for tag, side, storage, ptr, off, pad, snapshot in (
            ("ref", a, storage_a, ptr_a, off_a, pad_a, snap_a),
            ("res", b, storage_b, ptr_b, off_b, pad_b, snap_b)):
        cache = side["kv_cache"]
        # In-place write: the cache tensor, its backing storage and its offset
        # must be identical to the pre-call objects.
        assert cache.data_ptr() == ptr, tag
        assert cache.storage_offset() == off, tag
        assert cache._base is storage, tag
        assert torch.equal(cache, expected), tag
        if row_pad:
            # Cache columns beyond kv_lora_rank + pe_dim are padding.
            assert torch.equal(storage[..., width:], pad), tag
        # Inputs and the scale tensor are read-only.
        for name, saved in snapshot.items():
            assert torch.equal(side[name], saved), f"{tag}:{name} modified"

    ref_time = compute_ref_performance(**a)
    res_time = compute_res_performance(**b)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
