"""Accuracy and performance for v1_replace_nonfinite_inplace (DeepSpeed).

The official ``mask_nan_or_inf_with_val_inplace`` is an in-place helper for
gradient norms: NaN and +/-Inf become -1.0 regardless of the ``val`` argument,
finite values are untouched, and the function returns None.  The test observes
the return value, the mutated storage (same data_ptr) and every element.
"""
import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (value tuple, dtype, val): flat cases, including an all-finite control.
COMBOS = [
    (case, dtype, val)
    for case in (
        (float("nan"), float("inf"), -float("inf"), 0.0, -3.0, 4.0),
        (0.0, 1.0, -2.0),
        (float("nan"), 7.0, float("inf"), -float("inf")),
    )
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for val in (-1.0, 9.0)
]
COMBOS += [((((float("nan"), 2.0), (float("inf"), -4.0))), dtype, -1.0)
           for dtype in (torch.float32, torch.bfloat16)]


def input_build(config):
    case, dtype, val = config
    return {
        "input": torch.tensor(case, dtype=dtype),
        "device": None,
        "val": val,
    }


def _to_device(inputs):
    return {
        "input": inputs["input"].to(kernelgenbench.device),
        "device": kernelgenbench.device,
        "val": inputs["val"],
    }


def compute_ref(input, device=None, val=-1.0):
    return kernelgenbench.baseline.v1_replace_nonfinite_inplace(input, device, val)


def compute_res(input, device=None, val=-1.0):
    return kernelgenbench.solution.v1_replace_nonfinite_inplace(input, device, val)


def compare(ref_out, res_out, ref_input, res_input, original, val):
    assert ref_out is None and res_out is None
    assert res_input.shape == original.shape and res_input.dtype == original.dtype
    torch.testing.assert_close(res_input, ref_input, rtol=0, atol=0)
    # Values are the documented sentinel, never the caller's val.
    nonfinite = ~torch.isfinite(original)
    if bool(nonfinite.any()):
        assert bool((res_input.cpu()[nonfinite] == -1.0).all())
    assert bool((res_input.cpu()[~nonfinite] == original[~nonfinite]).all())


def compute_ref_performance(input, device=None, val=-1.0, initial=None):
    # The operator mutates its argument, so restore the pre-call values outside
    # the timed region to keep every repetition identical.
    def run():
        input.copy_(initial)
        return compute_ref(input, device, val)

    return _bench(run)


def compute_res_performance(input, device=None, val=-1.0, initial=None):
    def run():
        input.copy_(initial)
        return compute_res(input, device, val)

    return _bench(run)


def _bench(fn):
    import triton.testing

    return triton.testing.do_bench(fn, warmup=25, rep=100)


@label("v1_replace_nonfinite_inplace")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_replace_nonfinite_inplace(config):
    case, dtype, val = config
    inputs = input_build(config)
    initial = inputs["input"].clone()
    assert inputs["input"].dtype == dtype and not inputs["input"].is_cuda

    ref_inputs = _to_device(inputs)
    res_inputs = _to_device(inputs)
    ref_host = ref_inputs["input"]
    res_host = res_inputs["input"]
    ref_ptr = ref_host.data_ptr()
    res_ptr = res_host.data_ptr()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_host, res_host, initial, val)
    torch.testing.assert_close(ref_host.cpu(), res_host.cpu(), rtol=0, atol=0)
    # The update happened in the original storage, not in a fresh tensor.
    assert ref_host.data_ptr() == ref_ptr
    assert res_host.data_ptr() == res_ptr

    ref_time = compute_ref_performance(
        ref_inputs["input"], ref_inputs["device"], ref_inputs["val"], initial
    )
    res_time = compute_res_performance(
        res_inputs["input"], res_inputs["device"], res_inputs["val"], initial
    )
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
