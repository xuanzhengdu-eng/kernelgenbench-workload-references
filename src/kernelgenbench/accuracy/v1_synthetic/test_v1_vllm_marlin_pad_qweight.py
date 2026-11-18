"""Accuracy and performance for v1_vllm_marlin_pad_qweight (row 379).

Official source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L251-L261

``qweight`` is GPTQ-packed int32 ``[size_k / pack, size_n]`` with
``pack = size_k // qweight.size(0)``. If ``(padded_n, padded_k)`` already equals
``(size_n, size_k)`` the *same object* is returned; otherwise a new zero tensor of
``[size_k / pack + (padded_k - size_k) / pack, padded_n]`` is returned with the
input copied into its top-left corner. The input is never written.

This is its own item: upstream ``marlin_pad_scales`` (the separate row-379
reference) derives its row count from ``padded_k / group_size`` instead of the
pack factor, so the two entry points cannot share one contract.

Each COMBOS entry is ``(size_n, size_k, padded_n, padded_k)``.
"""
import statistics
import time

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

NAME = "v1_vllm_marlin_pad_qweight"

COMBOS = [
    (5, 16, 5, 16),
    (5, 16, 5, 24),
    (5, 16, 8, 16),
    (5, 16, 8, 32),
    (7, 32, 8, 48),
    (3, 16, 4, 16),
]


def input_build(config):
    size_n, size_k, padded_n, padded_k = config
    generator = torch.Generator().manual_seed(4300 + size_n + size_k)
    rows = size_k // 8
    values = torch.randint(0, 2**31 - 1, (rows, size_n), generator=generator, dtype=torch.int64)
    return {
        "qweight": values.to(torch.int32) + 1,
        "size_n": size_n,
        "size_k": size_k,
        "padded_n": padded_n,
        "padded_k": padded_k,
    }


def _fresh(inputs, device):
    return {
        "qweight": inputs["qweight"].to(device=device, copy=True),
        "size_n": inputs["size_n"],
        "size_k": inputs["size_k"],
        "padded_n": inputs["padded_n"],
        "padded_k": inputs["padded_k"],
    }


def compute_ref(qweight, size_n, size_k, padded_n, padded_k):
    return kernelgenbench.baseline.v1_vllm_marlin_pad_qweight(
        qweight, size_n, size_k, padded_n, padded_k
    )


def compute_res(qweight, size_n, size_k, padded_n, padded_k):
    return kernelgenbench.solution.v1_vllm_marlin_pad_qweight(
        qweight, size_n, size_k, padded_n, padded_k
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
def test_accuracy_v1_vllm_marlin_pad_qweight(config):
    size_n, size_k, padded_n, padded_k = config
    inputs = input_build(config)
    ref_inputs = _fresh(inputs, kernelgenbench.device)
    res_inputs = _fresh(inputs, kernelgenbench.device)
    before = res_inputs["qweight"].clone()
    identity = (padded_n, padded_k) == (size_n, size_k)
    want_shape = (
        (size_k // 8, size_n)
        if identity
        else (size_k // 8 + (padded_k - size_k) // 8, padded_n)
    )

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)

    for out, side in ((ref_out, ref_inputs), (res_out, res_inputs)):
        assert isinstance(out, torch.Tensor) and out.dtype == torch.int32
        assert out.shape == want_shape and out.device == side["qweight"].device
        if identity:
            assert out is side["qweight"], "no-padding case must return the input object"
        else:
            assert out.is_contiguous()
            assert not torch._C._is_alias_of(out, side["qweight"])
            # Top-left corner is the input, every padded element is zero.
            torch.testing.assert_close(out[: size_k // 8, :size_n], side["qweight"], rtol=0, atol=0)
            rest = out.clone()
            rest[: size_k // 8, :size_n] = 0
            assert torch.count_nonzero(rest) == 0, "padding must be zeros"
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(before, res_inputs["qweight"], rtol=0, atol=0)

    ref_time = _measure(compute_ref, config)
    res_time = _measure(compute_res, config)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
