"""Accuracy and performance for vLLM SiluAndMulWithClamp (gate clamped before silu).

alpha/beta are construction-time scalars of the SAME interface, so they are
separate COMBOS rather than separate items.  The test also shows that a gate
above the limit produces ``clamp*sigmoid(alpha*clamp)`` and NOT the
clamp-after-activation result of SwigluStepAndMul, which pins the clamp
position.
"""
import pytest
import torch
import torch.nn.functional as F

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows42_51_test_utils as u
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# (shape, dtype, clamp, alpha, beta): alpha/beta are numeric parameters of one
# interface; (1.702, 1.0) is the SwiGLU-OAI / MiniMax-M3 setting.
SHAPES = [(32, 256), (256, 2048)]
CLAMP = 1.5
COMBOS = [(shape, dtype, alpha, beta) for shape in SHAPES
          for dtype in (torch.float32, torch.float16, torch.bfloat16)
          for alpha, beta in ((1.0, 0.0), (1.702, 1.0))]

# The CUDA kernel evaluates the sigmoid in fp32 while the readable formula keeps
# the activation dtype, so allow a couple of ulps.
KERNEL_TOLERANCE = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (3e-2, 3e-2),
}


def input_build(config):
    shape, dtype, alpha, beta = config
    generator = torch.Generator(device="cpu").manual_seed(562393570776)
    packed = torch.randn(shape, dtype=dtype, generator=generator) * 2.0
    half = shape[-1] // 2
    packed[0, 0] = CLAMP + 5.0     # before-silu clamp must saturate here
    packed[0, 1] = CLAMP           # boundary
    packed[0, half] = -CLAMP - 3.0
    packed[0, half + 1] = CLAMP
    return {"y": packed, "clamp_value": CLAMP, "alpha": alpha, "beta": beta}


def compute_ref(y, clamp_value, alpha=1.0, beta=0.0):
    return kernelgenbench.baseline.v1_clamped_swiglu_vllm_pre(y, clamp_value, alpha=alpha, beta=beta)


def compute_res(y, clamp_value, alpha=1.0, beta=0.0):
    return kernelgenbench.solution.v1_clamped_swiglu_vllm_pre(y, clamp_value, alpha=alpha, beta=beta)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = KERNEL_TOLERANCE[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return u.measure(compute_ref, u.fresh(inputs, kernelgenbench.device))


def compute_res_performance(**inputs):
    return u.measure(compute_res, u.fresh(inputs, kernelgenbench.device))


@label("v1_clamped_swiglu_vllm_pre")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_clamped_swiglu_vllm_pre(config):
    shape, dtype, alpha, beta = config
    inputs = input_build(config)
    ref_inputs = u.fresh(inputs, kernelgenbench.device)
    res_inputs = u.fresh(inputs, kernelgenbench.device)
    before = u.clone_tensors(res_inputs)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    half = shape[-1] // 2
    rtol, atol = KERNEL_TOLERANCE[dtype]
    gate_clamped = torch.tensor(CLAMP, device=res_out.device)
    value0 = res_inputs["y"][0, half].float().clamp(-CLAMP, CLAMP)
    expected0 = gate_clamped * torch.sigmoid(alpha * gate_clamped) * (value0 + beta)
    torch.testing.assert_close(res_out[0, 0].float(), expected0.float(), rtol=rtol, atol=atol)

    # Clamp position: the same saturating input under the clamp-after-activation
    # formula (SwigluStepAndMul) gives limit*value, which differs here.
    post_formula = gate_clamped * value0
    assert not torch.allclose(res_out[0, 0].float(), post_formula.float(), rtol=1e-2, atol=1e-2), \
        "clamp-before-silu must differ from clamp-after-silu for a saturating gate"

    u.assert_unchanged(before, res_inputs)
    u.assert_fresh(res_out, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))


def test_pre_rejects_odd_packed_axis():
    with pytest.raises(ValueError):
        compute_ref(torch.randn(8, 255), CLAMP)
