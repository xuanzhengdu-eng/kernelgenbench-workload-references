"""Accuracy/performance for v1_batch_invariant_mean_megatron (Megatron operator).

Megatron's contract differs from vLLM's item: ``dim`` must be non-empty; a single
dimension is reduced by the deterministic ``mean_dim`` kernel and keeps the input dtype
even when ``dtype=torch.float32`` is passed (the argument is only validated); a
multi-dimension reduction sums in FP32 and always returns FP32.  That dtype difference is
what separates this Item from ``v1_batch_invariant_mean``.
"""
import pytest
import torch
import triton.testing

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

COMBOS = [
    ((1,), False, None, False, torch.float32),
    ((1,), True, torch.float32, False, torch.float16),
    ((1,), False, None, False, torch.bfloat16),
    ((0, 2), False, None, True, torch.float32),
    ((-1,), True, None, True, torch.float16),
    ((0,), False, None, False, torch.float32),
    ((0, 1), True, torch.float32, True, torch.bfloat16),
]


def input_build(config):
    dim, keepdim, dtype, strided, in_dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    tensor = torch.randn((3, 4, 10 if strided else 5), generator=generator,
                         dtype=in_dtype)
    return {"input": tensor, "dim": dim, "keepdim": keepdim, "dtype": dtype}


def prepare(inputs, device, strided=False):
    """Move first, then slice, so the last-dim stride really is 2 on the device."""
    moved = dict(inputs)
    moved["input"] = inputs["input"].to(device)
    if strided:
        moved["input"] = moved["input"][..., ::2]
    return moved


def compute_ref(input, dim, keepdim=False, dtype=None):
    return kernelgenbench.baseline.v1_batch_invariant_mean_megatron(
        input, dim, keepdim, dtype)


def compute_res(input, dim, keepdim=False, dtype=None):
    return kernelgenbench.solution.v1_batch_invariant_mean_megatron(
        input, dim, keepdim, dtype)


TOLERANCE = {
    torch.float32: (4e-6, 2e-6),
    torch.float16: (3e-3, 2e-3),
    torch.bfloat16: (2e-2, 1e-2),
}


def compare(ref_out, res_out, dtype=None):
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.device == ref_out.device
    rtol, atol = TOLERANCE[ref_out.dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    """Median GPU time of one reference call; warmup and device sync are inside do_bench."""
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    """Same measurement for the solution side, identical inputs and conditions."""
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_batch_invariant_mean_megatron")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_batch_invariant_mean_megatron(config):
    dim, keepdim, dtype, strided, in_dtype = config
    inputs = input_build(config)
    ref_inputs = prepare(inputs, kernelgenbench.device, strided)
    res_inputs = prepare(inputs, kernelgenbench.device, strided)
    original = res_inputs["input"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # Single-dimension reductions keep the input dtype; multi-dimension returns FP32.
    if len(dim) == 1:
        assert res_out.dtype == in_dtype, "single-dim mean must keep the input dtype"
    else:
        assert res_out.dtype == torch.float32, "multi-dim mean must return FP32"
    torch.testing.assert_close(res_inputs["input"], original, rtol=0, atol=0)

    if config is COMBOS[0]:
        # The extracted item guards a closed domain: an empty dim tuple is rejected even
        # though upstream would silently return sum(x)/1 for it (measured on the pinned
        # operator: 276.0 for arange(24) instead of the mean 11.5).  The reference must
        # not inherit that accidental behaviour.
        try:
            compute_ref(**(ref_inputs | {"dim": ()}))
        except AssertionError:
            pass
        else:
            raise AssertionError("reference must reject an empty dim tuple")

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
