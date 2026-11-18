"""Accuracy and performance for v1_vllm_marlin_pad_scales (row 379).

Official source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L264-L278

``scales`` is ``[num_groups, size_n]`` with one row per ``group_size`` K
elements. If ``(padded_n, padded_k) == (size_n, size_k)`` the same object is
returned. Otherwise the padded row count is
``padded_k / group_size - num_groups`` for a positive group size and 0 for a
channel-wise (``group_size <= 0``) weight; upstream asserts the delta is not
negative. The result is a new zero tensor with the input in its top-left corner;
the input is never written.

This is its own item: the row count comes from the quantization group size, not
from the packed-word factor used by the row-379 ``marlin_pad_qweight`` item.

Each COMBOS entry is ``(size_n, size_k, padded_n, padded_k, group_size)``.
"""
import statistics
import time

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

NAME = "v1_vllm_marlin_pad_scales"

COMBOS = [
    (5, 16, 5, 16, 4),
    (5, 16, 8, 16, 4),
    (5, 16, 8, 24, 4),
    (5, 16, 8, 32, 4),
    (5, 16, 8, 32, -1),
    (6, 24, 8, 32, 8),
    (7, 32, 8, 48, 8),
]


def input_build(config):
    size_n, size_k, padded_n, padded_k, group_size = config
    rows = size_k // group_size if group_size > 0 else 1
    generator = torch.Generator().manual_seed(4350 + size_n + size_k + group_size)
    values = torch.rand((rows, size_n), generator=generator, dtype=torch.float32)
    return {
        "scales": values,
        "size_n": size_n,
        "size_k": size_k,
        "padded_n": padded_n,
        "padded_k": padded_k,
        "group_size": group_size,
    }


def _fresh(inputs, device):
    out = {name: value for name, value in inputs.items() if not torch.is_tensor(value)}
    out["scales"] = inputs["scales"].to(device=device, copy=True)
    return out


def compute_ref(scales, size_n, size_k, padded_n, padded_k, group_size):
    return kernelgenbench.baseline.v1_vllm_marlin_pad_scales(
        scales, size_n, size_k, padded_n, padded_k, group_size
    )


def compute_res(scales, size_n, size_k, padded_n, padded_k, group_size):
    return kernelgenbench.solution.v1_vllm_marlin_pad_scales(
        scales, size_n, size_k, padded_n, padded_k, group_size
    )


def _measure(fn, config):
    device = kernelgenbench.device
    samples = []
    for _ in range(9):
        arguments = _fresh(input_build(config), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn(**arguments)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples[1:])


@label(NAME)
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_marlin_pad_scales(config):
    size_n, size_k, padded_n, padded_k, group_size = config
    inputs = input_build(config)
    ref_inputs = _fresh(inputs, kernelgenbench.device)
    res_inputs = _fresh(inputs, kernelgenbench.device)
    before = res_inputs["scales"].clone()
    identity = (padded_n, padded_k) == (size_n, size_k)
    pad_rows = padded_k // group_size - inputs["scales"].size(0) if group_size > 0 else 0
    want_shape = (
        (inputs["scales"].size(0), size_n)
        if identity
        else (inputs["scales"].size(0) + pad_rows, padded_n)
    )

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)

    for out, side in ((ref_out, ref_inputs), (res_out, res_inputs)):
        assert isinstance(out, torch.Tensor) and out.dtype == torch.float32
        assert out.shape == want_shape and out.device == side["scales"].device
        if identity:
            assert out is side["scales"], "no-padding case must return the input object"
        else:
            assert out.is_contiguous()
            assert not torch._C._is_alias_of(out, side["scales"])
            rows, cols = side["scales"].shape
            torch.testing.assert_close(out[:rows, :cols], side["scales"], rtol=0, atol=0)
            rest = out.clone()
            rest[:rows, :cols] = 0
            assert torch.count_nonzero(rest) == 0, "padding must be zeros"
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(before, res_inputs["scales"], rtol=0, atol=0)

    ref_time = _measure(compute_ref, config)
    res_time = _measure(compute_res, config)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
