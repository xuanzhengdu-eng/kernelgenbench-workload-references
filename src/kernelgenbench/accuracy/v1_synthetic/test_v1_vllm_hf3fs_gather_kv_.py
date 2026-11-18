"""Accuracy and performance for v1_vllm_hf3fs_gather_kv_.

Row 543 of the original 723-item table, item OP-f6c953bb40c6.  The official
Triton kernel (``kv_cache_gather_kernel`` behind ``gather_kv_caches``) is bound
as the Solution; the reference is the same contract written with tensor
indexing.

Contract under test
-------------------
``dst_tensor`` is ``[L, H, N, C]`` and every layer cache is a ``[B, H, T, C]``
view; the cache may be a permuted (non-contiguous) view, which is why upstream
reads it through the four element strides.  For each ``token position p`` with
``index i``:

* ``i < total_token_in_kvcache``: ``dst[l, :, p, :] = cache[l, i // T, :, i % T, :]``
* otherwise the destination row keeps whatever it held before.

The function mutates ``dst_tensor`` in place, returns ``None``, and never writes
the caches.  Every layer uses the same index list.  The empty index list is a
legal call (no rows are written).

COMBOS axes: (layers, blocks, heads, tokens_per_block, content, real token
total, token index list, dtype, cache layout).  ``transposed`` builds the cache
as ``[B, T, H, C]`` then permutes to ``[B, H, T, C]``, so its strides differ
from the contiguous layout while the logical shape is the same.
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
    (1, 2, 1, 8, 16, 15, (14, 0, 7, 7), torch.float16, "transposed"),
    (2, 2, 2, 4, 5, 8, (), torch.float32, "contiguous"),
    (1, 1, 1, 1, 1, 1, (0,), torch.float32, "transposed"),
    (4, 5, 2, 8, 33, 40, (39, 0, 20, 19, 38, 5), torch.float16, "contiguous"),
]


def input_build(config):
    """Deterministic CPU inputs for one config.

    ``kv_caches``: ``L`` tensors of logical shape ``[blocks, heads,
    tokens_per_block, content]``; the ``transposed`` variant is the same data
    built as ``[blocks, tokens_per_block, heads, content]`` and permuted, so
    ``is_contiguous()`` is False.  ``dst_tensor``: ``[L, heads, len(indices),
    content]`` filled with the sentinel ``-5``.  ``token_indices`` is a plain
    Python list of ints, as upstream expects for ``async_tensor_h2d``.
    """
    (layers, blocks, heads, tokens_per_block, content, total, indices,
     dtype, layout) = config
    caches = []
    for layer in range(layers):
        count = blocks * heads * tokens_per_block * content
        if layout == "contiguous":
            values = torch.arange(count, dtype=torch.float32).reshape(
                blocks, heads, tokens_per_block, content)
        else:
            # Same logical tensor stored as [blocks, tokens, heads, content] and
            # permuted into the [blocks, heads, tokens, content] view, so the
            # element strides are not the row-major ones.
            values = torch.arange(count, dtype=torch.float32).reshape(
                blocks, tokens_per_block, heads, content).permute(0, 2, 1, 3)
        values = (values + 31.0 * layer).remainder(251).to(dtype)
        assert values.shape == (blocks, heads, tokens_per_block, content)
        if layout == "contiguous":
            assert values.is_contiguous()
        elif heads > 1 and tokens_per_block > 1:
            # Degenerate (size-1) axes stay contiguous after the permute.
            assert not values.is_contiguous(), "transposed view must be strided"
        caches.append(values)
    dst = torch.full((layers, heads, len(indices), content), -5.0, dtype=dtype)
    return {
        "kv_caches": caches,
        "total_token_in_kvcache": total,
        "dst_tensor": dst,
        "token_indices": list(indices),
        "tokens_per_block": tokens_per_block,
    }


def _fresh_tensor(value, device):
    """Independent copy on ``device`` that keeps the original element strides."""
    target = torch.empty_strided(value.shape, value.stride(),
                                 dtype=value.dtype, device=device)
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
    """Expected destination before the call, built from the same CPU inputs."""
    (layers, blocks, heads, tokens_per_block, content, total, indices,
     dtype, layout) = config
    built = input_build(config)
    caches = built["kv_caches"]
    expected = built["dst_tensor"].to(device)
    for layer in range(layers):
        for position, index in enumerate(indices):
            if index >= total:
                continue
            expected[layer, :, position].copy_(
                caches[layer][index // tokens_per_block, :, index % tokens_per_block, :]
                .to(device))
    return expected


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_hf3fs_gather_kv_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_hf3fs_gather_kv_(**inputs)


def compare(ref_out, res_out, dtype, config, ref_inputs, res_inputs, expected):
    """Exact copy semantics plus in-place/read-only side effects."""
    assert ref_out is None and res_out is None, "gather returns None on both sides"
    assert ref_inputs["dst_tensor"].dtype == res_inputs["dst_tensor"].dtype == dtype
    # The gather is a pure copy: bit-exact, including sentinel rows for skipped
    # indices and padding rows outside ``token_indices``.
    torch.testing.assert_close(ref_inputs["dst_tensor"], expected, rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["dst_tensor"], expected, rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["dst_tensor"], ref_inputs["dst_tensor"],
                               rtol=0, atol=0)
    # Skip behaviour: indices >= total must leave their row at the sentinel.
    total = config[5]
    for position, index in enumerate(config[6]):
        if index >= total:
            row = res_inputs["dst_tensor"][:, :, position, :]
            torch.testing.assert_close(row, torch.full_like(row, -5.0), rtol=0, atol=0)


def measure(fn, inputs, device, repeats=20):
    """Median single-call latency in milliseconds, inputs rebuilt outside the timer."""
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


@label("v1_vllm_hf3fs_gather_kv_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_hf3fs_gather_kv_(config):
    inputs = input_build(config)
    device = kernelgenbench.device
    ref_inputs = fresh(inputs, device)
    res_inputs = fresh(inputs, device)
    before_caches = [cache.clone() for cache in res_inputs["kv_caches"]]
    dst_pointer = res_inputs["dst_tensor"].data_ptr()
    expected = oracle(config, device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()

    compare(ref_out, res_out, config[7], config, ref_inputs, res_inputs, expected)
    assert dst_pointer == res_inputs["dst_tensor"].data_ptr(), "dst storage replaced"
    for before, after in zip(before_caches, res_inputs["kv_caches"]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    ref_time = compute_ref_performance(inputs, device)
    res_time = compute_res_performance(inputs, device)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
