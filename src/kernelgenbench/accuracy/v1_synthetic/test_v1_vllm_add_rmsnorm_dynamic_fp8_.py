"""Accuracy and performance for v1_vllm_add_rmsnorm_dynamic_fp8_ (vLLM dynamic)."""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (rows, hidden, scale_upper_bound, zero_residual_sum).  hidden=35 keeps the
# non-vectorized kernel path; scale_ub exercises the amax clamp; the last case
# makes one row's residual exactly cancel the input.
COMBOS = [(1, 35, None, False), (3, 128, None, False), (4, 128, 0.5, False),
          (2, 64, None, True), (2, 256, 2.0, False), (2, 256, None, False)]


def input_build(config):
    m, h, ub, zero = config
    generator = torch.Generator(device="cpu").manual_seed(m * 1000 + h)
    inp = torch.randn(m, h, generator=generator)
    residual = torch.randn(m, h, generator=generator)
    weight = 0.5 + torch.rand(h, generator=generator)
    if zero:
        residual[0] = -inp[0]
    return {"inp": inp, "residual": residual, "weight": weight, "epsilon": 1e-5,
            "scale_ub": None if ub is None else torch.tensor(ub)}


def compute_ref(inp, residual, weight, epsilon, scale_ub):
    return kernelgenbench.baseline.v1_vllm_add_rmsnorm_dynamic_fp8_(
        inp, residual, weight, epsilon, scale_ub)


def compute_res(inp, residual, weight, epsilon, scale_ub):
    return kernelgenbench.solution.v1_vllm_add_rmsnorm_dynamic_fp8_(
        inp, residual, weight, epsilon, scale_ub)


def compare(ref_out, res_out, config, ref_residual, res_residual):
    m, h, _, _ = config
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    quant, scale = res_out
    assert quant.shape == (m, h) and quant.dtype == torch.float8_e4m3fn
    assert scale.shape == (m, 1) and scale.dtype == torch.float32
    assert torch.equal(quant, ref_out[0]), "vLLM dynamic quant must be bit-exact"
    torch.testing.assert_close(scale, ref_out[1], rtol=1e-5, atol=1e-6)
    # The residual buffer is updated in place with the FP32 sum cast back.
    assert torch.equal(res_residual, ref_residual)


def compute_ref_performance(inp, residual, weight, epsilon, scale_ub):
    return measure(compute_ref, {"inp": inp, "residual": residual, "weight": weight,
                                 "epsilon": epsilon, "scale_ub": scale_ub})


def compute_res_performance(inp, residual, weight, epsilon, scale_ub):
    return measure(compute_res, {"inp": inp, "residual": residual, "weight": weight,
                                 "epsilon": epsilon, "scale_ub": scale_ub})


@label("v1_vllm_add_rmsnorm_dynamic_fp8_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_add_rmsnorm_dynamic_fp8_(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_ptr = ref_inputs["residual"].data_ptr()
    res_ptr = res_inputs["residual"].data_ptr()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, ref_inputs["residual"], res_inputs["residual"])

    # Same residual storage is reused (in-place contract) and inp/weight are
    # untouched.
    assert ref_inputs["residual"].data_ptr() == ref_ptr
    assert res_inputs["residual"].data_ptr() == res_ptr
    for name in ("inp", "weight"):
        torch.testing.assert_close(ref_inputs[name].cpu(), inputs[name], rtol=0, atol=0)
        torch.testing.assert_close(res_inputs[name].cpu(), inputs[name], rtol=0, atol=0)
    assert not torch._C._is_alias_of(res_out[0], res_inputs["inp"])

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
