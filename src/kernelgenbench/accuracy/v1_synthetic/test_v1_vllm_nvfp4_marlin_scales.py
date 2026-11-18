"""Accuracy and performance for v1_vllm_nvfp4_marlin_scales.

COMBOS cover an explicit factor equal to one, a rescaling factor above one, a
factor below one (which the upstream code keeps without rescaling) and the
automatic factor computed by upstream when ``scale_factor`` is None.  The test
checks the two-tuple return, the returned factor, the byte-exact E4M3 scales,
the preserved shape and the no-write contract.
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

# (rows, columns, source dtype, explicit rescaling factor or None for automatic)
COMBOS = [
    (1, 8, torch.float16, 1.0),
    (2, 8, torch.float32, 2.0),
    (2, 12, torch.bfloat16, 0.5),
    (3, 8, torch.float32, None),
    (2, 16, torch.float16, None),
]


def input_build(config):
    rows, cols, dtype, factor = config
    values = (torch.arange(rows * cols, dtype=torch.float32) % 13) / 64
    return {"marlin_scales": values.to(dtype).reshape(rows, cols),
            "scale_factor": factor}


def compute_ref(marlin_scales, scale_factor):
    return kernelgenbench.baseline.v1_vllm_nvfp4_marlin_scales(
        marlin_scales, scale_factor)


def compute_res(marlin_scales, scale_factor):
    return kernelgenbench.solution.v1_vllm_nvfp4_marlin_scales(
        marlin_scales, scale_factor)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, tuple) and len(res_out) == 2, (
        "vLLM variant returns (scales, scale_factor)")
    assert ref_out[1] == res_out[1], "scale_factor must be returned unchanged"
    ref_scales, res_scales = ref_out[0], res_out[0]
    assert res_scales.dtype == torch.float8_e4m3fn
    assert res_scales.shape == ref_scales.shape
    assert res_scales.is_contiguous()
    assert torch.equal(bytes_of(res_scales), bytes_of(ref_scales)), "E4M3 scale bytes differ"


def _args(marlin_scales, factor):
    def build():
        return {"marlin_scales": marlin_scales.clone(), "scale_factor": factor}
    return build


def compute_ref_performance(marlin_scales, scale_factor):
    marlin_scales = marlin_scales.to(kernelgenbench.device)
    return measure(compute_ref, None, reset=_args(marlin_scales, scale_factor))


def compute_res_performance(marlin_scales, scale_factor):
    marlin_scales = marlin_scales.to(kernelgenbench.device)
    return measure(compute_res, None, reset=_args(marlin_scales, scale_factor))


@label("v1_vllm_nvfp4_marlin_scales")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_nvfp4_marlin_scales(config):
    rows, cols, dtype, factor = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before = res_inputs["marlin_scales"].clone()
    assert res_inputs["marlin_scales"].shape == (rows, cols)
    assert res_inputs["marlin_scales"].dtype == dtype

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    scales = res_out[0]
    assert scales.shape == (rows, cols)
    assert res_out[1] == (factor if factor is not None else ref_out[1])
    if factor is None:
        assert res_out[1] >= 1.0, "automatic factor is a power of two >= 1"
        assert float(res_out[1]) == float(ref_out[1])
    assert not torch._C._is_alias_of(scales, res_inputs["marlin_scales"])
    assert_unchanged({"marlin_scales": before}, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
