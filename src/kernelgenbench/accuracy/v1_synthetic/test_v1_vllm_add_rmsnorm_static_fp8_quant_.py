"""Accuracy and performance for v1_vllm_add_rmsnorm_static_fp8_quant_ (vLLM static)."""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (rows, hidden, dtype): the static kernel requires residual/weight in the input
# dtype and writes FP8 into the caller-provided ``out`` buffer.
COMBOS = [(2, 32, torch.float32), (3, 64, torch.bfloat16),
          (2, 48, torch.float16), (4, 128, torch.bfloat16)]


def input_build(config):
    m, h, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(719 + m * 31 + h)
    inp = (torch.randn(m, h, generator=generator) * 0.5).to(dtype)
    weight = (0.5 + torch.rand(h, generator=generator)).to(dtype)
    residual = (torch.randn(m, h, generator=generator) * 0.25).to(dtype)
    return {"inp": inp, "residual": residual, "weight": weight, "epsilon": 1e-5,
            "static_scale": torch.tensor([0.02], dtype=torch.float32),
            "out": torch.empty((m, h), dtype=torch.float8_e4m3fn)}


def compute_ref(inp, residual, weight, epsilon, static_scale, out):
    return kernelgenbench.baseline.v1_vllm_add_rmsnorm_static_fp8_quant_(
        inp, residual, weight, epsilon, static_scale, out)


def compute_res(inp, residual, weight, epsilon, static_scale, out):
    return kernelgenbench.solution.v1_vllm_add_rmsnorm_static_fp8_quant_(
        inp, residual, weight, epsilon, static_scale, out)


def compare(ref_out, res_out, config, ref_inputs, res_inputs):
    m, h, _ = config
    # Contract: returns None and writes the FP8 result into the provided buffer.
    assert ref_out is None and res_out is None
    assert torch.equal(res_inputs["out"], ref_inputs["out"]), "static quant must be bit-exact"
    assert res_inputs["out"].dtype == torch.float8_e4m3fn
    assert res_inputs["out"].shape == (m, h)
    assert torch.equal(res_inputs["residual"], ref_inputs["residual"])


def compute_ref_performance(inp, residual, weight, epsilon, static_scale, out):
    return measure(compute_ref, {"inp": inp, "residual": residual, "weight": weight,
                                 "epsilon": epsilon, "static_scale": static_scale, "out": out})


def compute_res_performance(inp, residual, weight, epsilon, static_scale, out):
    return measure(compute_res, {"inp": inp, "residual": residual, "weight": weight,
                                 "epsilon": epsilon, "static_scale": static_scale, "out": out})


@label("v1_vllm_add_rmsnorm_static_fp8_quant_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_add_rmsnorm_static_fp8_quant_(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out_ptr = ref_inputs["out"].data_ptr()
    res_out_ptr = res_inputs["out"].data_ptr()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, ref_inputs, res_inputs)

    # The same ``out`` and ``residual`` storage must be reused.
    assert ref_inputs["out"].data_ptr() == ref_out_ptr
    assert res_inputs["out"].data_ptr() == res_out_ptr
    for name in ("inp", "weight", "static_scale"):
        torch.testing.assert_close(ref_inputs[name].cpu(), inputs[name], rtol=0, atol=0)
        torch.testing.assert_close(res_inputs[name].cpu(), inputs[name], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
