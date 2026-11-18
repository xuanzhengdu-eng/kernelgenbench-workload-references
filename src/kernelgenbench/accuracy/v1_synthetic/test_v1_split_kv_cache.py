"""Accuracy and performance for v1_split_kv_cache.

Input: a DeepSpeed inference-v2 KV cache shaped [blocks, block_size, 2, heads,
head_size]. Output: the key and value subtensors [blocks, block_size, heads,
head_size] as views into the original storage.

Key assertions: tuple length, shapes/dtypes, bitwise values, identical strides
and storage offsets, storage aliasing with the input cache for both outputs,
and an untouched input. A paged-stride cache (every second block of a larger
buffer) covers a non-contiguous backing storage.
"""
import statistics

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

NAME = "v1_split_kv_cache"

# (blocks, block_size, hidden), dtype, storage layout
COMBOS = [
    (shape, dtype, layout)
    for shape in [(2, 16, 128), (2, 128, 1024)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for layout in ("contiguous", "paged_stride")
]


def input_build(config):
    (blocks, block_size, hidden), dtype, layout = config
    head_size = hidden // 4
    generator = torch.Generator().manual_seed(20261109)
    if layout == "contiguous":
        cache = torch.randn((blocks, block_size, 2, 4, head_size),
                            dtype=dtype, generator=generator)
    else:
        backing = torch.randn((blocks, 2 * block_size, 2, 4, head_size),
                              dtype=dtype, generator=generator)
        cache = backing[:, ::2]
    return {"kv_cache": cache}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_split_kv_cache(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_split_kv_cache(**inputs)


def compare(ref_out, res_out, kv_cache, config):
    (blocks, block_size, hidden), dtype, _ = config
    head_size = hidden // 4
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 2
    for index, (expected, actual) in enumerate(zip(ref_out, res_out)):
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == expected.shape == (blocks, block_size, 4, head_size)
        assert actual.dtype == expected.dtype == dtype
        assert torch.equal(actual, expected), f"part {index} values differ"
        assert actual.stride() == expected.stride(), f"part {index} stride differs"
        assert actual.storage_offset() == expected.storage_offset()
        assert actual.untyped_storage().data_ptr() == kv_cache.untyped_storage().data_ptr(), \
            f"part {index} must be a view of the cache storage"
    assert ref_out[0].storage_offset() != ref_out[1].storage_offset(), \
        "key and value views must start at different offsets"


def _to_device(tensor, device):
    """Move a tensor keeping its (possibly padded) strides; .to() densifies."""
    if tensor.is_contiguous():
        return tensor.to(device)
    moved = torch.empty_strided(tuple(tensor.shape), tuple(tensor.stride()),
                                dtype=tensor.dtype, device=device)
    moved.copy_(tensor)
    return moved


def _bench(fn, inputs):
    device = kernelgenbench.device
    prepared = {"kv_cache": _to_device(inputs["kv_cache"], device)}
    for _ in range(10):
        fn(**prepared)
    torch.cuda.synchronize()
    samples = []
    for _ in range(50):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(**prepared)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def compute_ref_performance(inputs):
    return _bench(compute_ref, inputs)


def compute_res_performance(inputs):
    return _bench(compute_res, inputs)


@label("v1_split_kv_cache")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_split_kv_cache(config):
    inputs = input_build(config)
    cache = _to_device(inputs["kv_cache"], kernelgenbench.device)
    before = cache.clone()
    ref_out = compute_ref(kv_cache=cache)
    res_out = compute_res(kv_cache=cache)
    compare(ref_out, res_out, cache, config)
    assert torch.equal(before, cache), "KV cache was modified"
    if config[2] == "paged_stride":
        assert not cache.is_contiguous(), "paged_stride case must stay non-contiguous"

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
