"""Accuracy and performance for v1_sgl_nvfp4_marlin_scales.

COMBOS cover float16/float32/bfloat16 sources.  Scales are non-negative and
small so the FP16 intermediate, the 2**7 multiply and the E4M3 byte
extraction are exactly representable.  The test checks the single-tensor
return, the byte-exact E4M3 scales, the preserved shape and the no-write
contract; the vLLM tuple-returning variant is a separate item and test.
"""
import kernelgenbench
import pytest
import torch
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows352_361_utils import (
    assert_unchanged,
    bytes_of,
    fresh,
    measure,
)

# (rows, columns, source dtype)
COMBOS = [
    (1, 8, torch.float16),
    (2, 8, torch.float32),
    (2, 12, torch.bfloat16),
    (3, 8, torch.float32),
]


def input_build(config):
    rows, cols, dtype = config
    values = (torch.arange(rows * cols, dtype=torch.float32) % 13) / 64
    return {"marlin_scales": values.to(dtype).reshape(rows, cols)}


def compute_ref(marlin_scales):
    return kernelgenbench.baseline.v1_sgl_nvfp4_marlin_scales(marlin_scales)


def compute_res(marlin_scales):
    return kernelgenbench.solution.v1_sgl_nvfp4_marlin_scales(marlin_scales)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, torch.Tensor), "SGLang variant returns one tensor"
    assert res_out.dtype == torch.float8_e4m3fn
    assert res_out.shape == ref_out.shape
    assert res_out.is_contiguous()
    assert torch.equal(bytes_of(res_out), bytes_of(ref_out)), "E4M3 scale bytes differ"


def _args(marlin_scales):
    def build():
        return {"marlin_scales": marlin_scales.clone()}
    return build


def compute_ref_performance(marlin_scales):
    marlin_scales = marlin_scales.to(kernelgenbench.device)
    return measure(compute_ref, None, reset=_args(marlin_scales))


def compute_res_performance(marlin_scales):
    marlin_scales = marlin_scales.to(kernelgenbench.device)
    return measure(compute_res, None, reset=_args(marlin_scales))


@label("v1_sgl_nvfp4_marlin_scales")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_nvfp4_marlin_scales(config):
    rows, cols, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before = res_inputs["marlin_scales"].clone()
    assert res_inputs["marlin_scales"].shape == (rows, cols)
    assert res_inputs["marlin_scales"].dtype == dtype

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    # int16 -> two E4M3 bytes doubles the last dimension, then [:,1::2]
    # halves it again, so the public scale tensor keeps the input shape.
    assert res_out.shape == (rows, cols)
    assert not torch._C._is_alias_of(res_out, res_inputs["marlin_scales"])
    assert_unchanged({"marlin_scales": before}, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
