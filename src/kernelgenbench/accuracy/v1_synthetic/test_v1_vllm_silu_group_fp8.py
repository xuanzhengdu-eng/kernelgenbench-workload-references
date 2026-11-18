"""vLLM SiLU-and-mul + per-token group FP8 quantization; row 23, OP-5fb4b5bf8529.

The tuple contract is (fp8 output [M, H], fp32 scales [M, H/group] with (1, M)
strides). Cases cover fp32/fp16/bf16, UE8M0 power-of-two scales, clamp, alpha and
beta, a supplied output buffer (pointer preservation and in-place write), a
zero input (scales must still be finite) and both group sizes used by the old
test. The solution binds vLLM's Triton function, so the real kernel executes.
"""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.activation_rows22_31_checks import (
    config_dtype, fresh, measure, timings,
)
from sandbox.verifier.test_parametrize import label, parametrize


# (M, H, group, dtype, use_ue8m0, clamp_limit, alpha, beta, reuse_output, zero)
# M % group == 0 and H % group == 0; input is [M, 2H] and output is [M, H].
CASES = [
    (32, 64, 32, torch.float32, False, None, 1.0, 0.0, False, False),
    (32, 64, 32, torch.bfloat16, True, 2.5, 1.702, 1.0, True, False),
    (128, 128, 128, torch.bfloat16, False, None, 1.0, 0.0, False, False),
    (32, 64, 32, torch.float32, True, None, 1.0, 0.0, True, True),
    (128, 128, 128, torch.float16, False, None, 1.0, 0.0, False, False),
]
COMBOS = CASES


def input_build(config):
    """input [M, 2H]; output is an empty [M, H] fp8 buffer when reuse is set."""
    m, h, group, dtype, ue, clamp, alpha, beta, reuse, zero = config
    rng = torch.Generator(device="cpu").manual_seed(412)
    x = torch.randn((m, 2 * h), dtype=dtype, generator=rng) * 1.2
    if zero:
        x.zero_()
    out = torch.empty((m, h), dtype=torch.float8_e4m3fn) if reuse else None
    return {
        "input": x,
        "output": out,
        "use_ue8m0": ue,
        "eps": 1e-10,
        "clamp_limit": clamp,
        "group_size": group,
        "alpha": alpha,
        "beta": beta,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_silu_group_fp8(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_silu_group_fp8(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    ref_q, ref_s = ref_out
    res_q, res_s = res_out
    assert res_q.dtype == torch.float8_e4m3fn
    assert res_s.dtype == torch.float32
    assert res_q.shape == ref_q.shape
    assert res_s.shape == ref_s.shape
    assert res_s.stride() == ref_s.stride()
    # One FP8 rounding step is allowed after the sigmoid approximation.
    torch.testing.assert_close(res_q.float(), ref_q.float(), rtol=0.125, atol=0.125)
    torch.testing.assert_close(res_s, ref_s, rtol=1e-6, atol=1e-9)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_vllm_silu_group_fp8")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_silu_group_fp8(config):
    m, h, group, dtype, ue, clamp, alpha, beta, reuse, zero = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["input"].shape == (m, 2 * h)
        assert side["input"].dtype == dtype

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # A supplied output buffer is written in place and returned; input is read-only.
    if reuse:
        assert res_out[0].data_ptr() == res_inputs["output"].data_ptr()
        assert ref_out[0].data_ptr() == ref_inputs["output"].data_ptr()
    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["input"].cpu(), inputs["input"], rtol=0, atol=0)
    assert res_out[1].shape == (m, h // group)
    assert res_out[1].stride() == (1, m)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
