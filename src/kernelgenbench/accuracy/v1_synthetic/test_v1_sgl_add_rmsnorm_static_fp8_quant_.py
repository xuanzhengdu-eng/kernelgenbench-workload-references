"""Accuracy and performance for v1_sgl_add_rmsnorm_static_fp8_quant_ (SGLang static)."""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (rows, hidden, dtype, has_residual, separate post-residual addend).
COMBOS = [(2, 32, torch.bfloat16, True, False), (3, 64, torch.float16, True, True),
          (2, 48, torch.bfloat16, False, False), (2, 256, torch.bfloat16, True, True),
          (4, 128, torch.float16, True, False)]


def input_build(config):
    m, h, dtype, has_res, post = config
    generator = torch.Generator(device="cpu").manual_seed(719 + m * 31 + h)
    inp = (torch.randn(m, h, generator=generator) * 0.5).to(dtype)
    weight = (0.5 + torch.rand(h, generator=generator)).to(dtype)
    residual = (torch.randn(m, h, generator=generator) * 0.25).to(dtype) if has_res else None
    extra = (torch.randn(m, h, generator=generator) * 0.1).to(dtype) if post else None
    return {"inp": inp, "residual": residual, "weight": weight, "epsilon": 1e-5,
            "static_scale": torch.tensor([0.02], dtype=torch.float32),
            "post_residual_addition": extra}


def compute_ref(inp, residual, weight, epsilon, static_scale, post_residual_addition):
    return kernelgenbench.baseline.v1_sgl_add_rmsnorm_static_fp8_quant_(
        inp, residual, weight, epsilon, static_scale, post_residual_addition)


def compute_res(inp, residual, weight, epsilon, static_scale, post_residual_addition):
    return kernelgenbench.solution.v1_sgl_add_rmsnorm_static_fp8_quant_(
        inp, residual, weight, epsilon, static_scale, post_residual_addition)


def compare(ref_out, res_out, config, ref_inputs, res_inputs):
    m, h, dtype, has_res, post = config
    if not has_res:
        ref_triple, res_triple = ref_out, res_out
    else:
        # With a residual the return is ((fp8, scale, orig_dtype), residual_out).
        assert isinstance(ref_out, tuple) and len(ref_out) == 2
        assert isinstance(res_out, tuple) and len(res_out) == 2
        ref_triple, ref_res = ref_out
        res_triple, res_res = res_out
        assert torch.equal(res_res, ref_res)
        if res_inputs["post_residual_addition"] is None:
            assert res_res.data_ptr() == res_inputs["residual"].data_ptr(), \
                "residual must be updated in place"
        else:
            # With a separate addend the fused kernel returns a new residual.
            assert not torch._C._is_alias_of(res_res, res_inputs["residual"])
    assert isinstance(res_triple, tuple) and len(res_triple) == 3
    quant, scale, orig_dtype = res_triple
    assert quant.shape == (m, h) and quant.dtype == torch.float8_e4m3fn
    assert scale is res_inputs["static_scale"]
    assert orig_dtype == dtype
    assert not torch._C._is_alias_of(quant, res_inputs["inp"])
    # The fused kernel multiplies by the reciprocal of the static scale and
    # rounds through the activation dtype, so allow one/two FP8 codes.
    torch.testing.assert_close(quant.float(), ref_triple[0].float(), rtol=1e-2, atol=4.0)


def compute_ref_performance(inp, residual, weight, epsilon, static_scale, post_residual_addition):
    return measure(compute_ref, {"inp": inp, "residual": residual, "weight": weight,
                                 "epsilon": epsilon, "static_scale": static_scale,
                                 "post_residual_addition": post_residual_addition})


def compute_res_performance(inp, residual, weight, epsilon, static_scale, post_residual_addition):
    return measure(compute_res, {"inp": inp, "residual": residual, "weight": weight,
                                 "epsilon": epsilon, "static_scale": static_scale,
                                 "post_residual_addition": post_residual_addition})


@label("v1_sgl_add_rmsnorm_static_fp8_quant_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_add_rmsnorm_static_fp8_quant_(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, ref_inputs, res_inputs)

    for name in ("inp", "weight", "static_scale"):
        torch.testing.assert_close(ref_inputs[name].cpu(), inputs[name], rtol=0, atol=0)
        torch.testing.assert_close(res_inputs[name].cpu(), inputs[name], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
