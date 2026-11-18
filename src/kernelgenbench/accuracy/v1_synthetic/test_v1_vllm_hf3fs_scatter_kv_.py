"""Accuracy and performance for v1_vllm_hf3fs_scatter_kv_.

Row 544 of the original 723-item table, item OP-fcff9dd5116c.  The official
Triton kernel (``kv_cache_scatter_kernel`` behind ``scatter_kv_caches``) is
bound as the Solution; the reference is the same contract written with tensor
indexing.

Contract under test
-------------------
``src_tensor`` is ``[L, H, N, C]`` and every layer cache is a ``[B, H, T, C]``
view that may be a permuted (non-contiguous) view, which is why upstream writes
through the four element strides.  For each ``token position p`` with
``index i``:

* ``i < total_token_in_kvcache``: ``cache[l, i // T, :, i % T, :] = src[l, :, p, :]``
* otherwise the cache slot is left untouched (sentinel ``-9`` stays).

The function mutates the caches in place, returns ``None``, and never writes
``src_tensor``.  Slots that no index selects stay at the sentinel, which is how
the "skipped index" and "index outside the token list" branches are observed.

COMBOS axes: (layers, blocks, heads, tokens_per_block, content, real token
total, token index list, dtype, cache layout).  ``transposed`` builds each cache
as ``[B, T, H, C]`` then permutes it into the ``[B, H, T, C]`` view, so the
element strides differ from the row-major ones while the logical shape matches.
"""
import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


COMBOS = [
    # layers, blocks, heads, tokens/block, content, total, indices, dtype, layout
    (2, 3, 2, 4, 7, 10, (0, 7, 9, 10, 20), torch.float32, "contiguous"),
    (3, 3, 3, 4, 65, 11, (11, 3, 9, 4, 1), torch.bfloat16, "transposed"),
    (1, 2, 1, 8, 16, 15, (14, 0, 7, 6), torch.float16, "transposed"),
    (2, 2, 2, 4, 5, 8, (), torch.float32, "contiguous"),
    (1, 1, 1, 1, 1, 1, (0,), torch.float32, "transposed"),
    (4, 5, 2, 8, 33, 40, (39, 0, 20, 19, 38, 5), torch.float16, "contiguous"),
]


def input_build(config):
    """Deterministic CPU inputs for one config.

    ``kv_caches``: ``L`` tensors of logical shape ``[blocks, heads,
    tokens_per_block, content]``, all filled with the sentinel ``-9``; the
    ``transposed`` variant is built as ``[blocks, tokens_per_block, heads,
    content]`` and permuted, so it is a strided view of the same logical shape.
    ``src_tensor``: ``[L, heads, len(indices), content]`` with distinct values
    per layer/head/position so a misplaced write is visible.
    """
    (layers, blocks, heads, tokens_per_block, content, total, indices,
     dtype, layout) = config
    caches = []
    for _ in range(layers):
        if layout == "contiguous":
            cache = torch.full((blocks, heads, tokens_per_block, content), -9.0,
                               dtype=dtype)
        else:
            cache = torch.full((blocks, tokens_per_block, heads, content), -9.0,
                               dtype=dtype).permute(0, 2, 1, 3)
        assert cache.shape == (blocks, heads, tokens_per_block, content)
        if layout == "contiguous":
            assert cache.is_contiguous()
        elif heads > 1 and tokens_per_block > 1:
            assert not cache.is_contiguous(), "transposed view must be strided"
        caches.append(cache)
    count = layers * heads * len(indices) * content
    src = (torch.arange(count, dtype=torch.float32).reshape(
        layers, heads, len(indices), content).remainder(127)).to(dtype)
    return {
        "kv_caches": caches,
        "total_token_in_kvcache": total,
        "src_tensor": src,
        "token_indices": list(indices),
        "tokens_per_block": tokens_per_block,
    }


def _fresh_tensor(value, device):
    target = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype,
                                 device=device)
    target.copy_(value)
    return target


def fresh(inputs, device):
    """Independent copy of every tensor (including the cache list), same layout."""
    out = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            out[name] = _fresh_tensor(value, device)
        elif isinstance(value, list):
            out[name] = [_fresh_tensor(item, device)
                         if isinstance(item, torch.Tensor) else item
                         for item in value]
        else:
            out[name] = value
    return out


def oracle(config, device):
    """Expected cache contents after the scatter."""
    (layers, blocks, heads, tokens_per_block, content, total, indices,
     dtype, layout) = config
    built = input_build(config)
    expected = [cache.to(device) for cache in built["kv_caches"]]
    src = built["src_tensor"].to(device)
    for layer in range(layers):
        for position, index in enumerate(indices):
            if index >= total:
                continue
            expected[layer][index // tokens_per_block, :, index % tokens_per_block, :].copy_(
                src[layer, :, position])
    return expected


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_hf3fs_scatter_kv_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_hf3fs_scatter_kv_(**inputs)


def compare(ref_out, res_out, dtype, config, ref_inputs, res_inputs, expected):
    """Exact write semantics, sentinel preservation and read-only source."""
    assert ref_out is None and res_out is None, "scatter returns None on both sides"
    assert len(res_inputs["kv_caches"]) == len(expected)
    for actual, want, ref_cache in zip(res_inputs["kv_caches"], expected,
                                       ref_inputs["kv_caches"]):
        assert actual.dtype == dtype and actual.shape == want.shape
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
        torch.testing.assert_close(ref_cache, want, rtol=0, atol=0)
    # Skipped indices and unselected slots keep the sentinel.
    total = config[5]
    for position, index in enumerate(config[6]):
        if index < total:
            continue
        block, slot = index // config[3], index % config[3]
        if block < config[1]:
            for cache in res_inputs["kv_caches"]:
                row = cache[block, :, slot, :]
                torch.testing.assert_close(row, torch.full_like(row, -9.0),
                                           rtol=0, atol=0)


def measure(fn, inputs, device, repeats=20):
    """Median single-call latency in milliseconds; inputs rebuilt outside the timer."""
    for _ in range(3):
        fn(**fresh(inputs, device))
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        arguments = fresh(inputs, device)
        if torch.device(device).type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(**arguments)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            import time
            begin = time.perf_counter()
            fn(**arguments)
            samples.append((time.perf_counter() - begin) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def compute_ref_performance(inputs, expected_device):
    return measure(compute_ref, inputs, expected_device)


def compute_res_performance(inputs, expected_device):
    return measure(compute_res, inputs, expected_device)


@label("v1_vllm_hf3fs_scatter_kv_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_hf3fs_scatter_kv_(config):
    inputs = input_build(config)
    device = kernelgenbench.device
    ref_inputs = fresh(inputs, device)
    res_inputs = fresh(inputs, device)
    ref_pointers = [cache.data_ptr() for cache in ref_inputs["kv_caches"]]
    res_pointers = [cache.data_ptr() for cache in res_inputs["kv_caches"]]
    source_before = res_inputs["src_tensor"].clone()
    expected = oracle(config, device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()

    compare(ref_out, res_out, config[7], config, ref_inputs, res_inputs, expected)
    # Cache objects/storages are reused in place, and the source is untouched.
    assert ref_pointers == [cache.data_ptr() for cache in ref_inputs["kv_caches"]]
    assert res_pointers == [cache.data_ptr() for cache in res_inputs["kv_caches"]]
    torch.testing.assert_close(res_inputs["src_tensor"], source_before, rtol=0, atol=0)

    ref_time = compute_ref_performance(inputs, device)
    res_time = compute_res_performance(inputs, device)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
