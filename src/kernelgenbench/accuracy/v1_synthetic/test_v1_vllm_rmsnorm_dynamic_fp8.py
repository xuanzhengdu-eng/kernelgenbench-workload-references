"""Accuracy and performance for v1_vllm_rmsnorm_dynamic_fp8 (row 64).

Contract: no-residual dynamic per-token RMSNorm quant for finite FP32 input.
Output is E4M3 (clamped to +-448) plus one fp32 scale per token; the scale is
max(amax/448, 1/(448*512)) and an optional scale_ub caps amax first.
Official: https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L435-L451
Kernel: https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu#L12-L97
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows62_71_checks import (
    fresh, measure, result, unchanged,
)
from sandbox.verifier.test_parametrize import label, parametrize

FP8_MAX = 448.0
MIN_SCALE = 1.0 / (448.0 * 512.0)

# (rows, hidden, scale_ub, zero_first_row); input and weight are float32.
COMBOS = [
    (1, 35, None, False),
    (3, 128, None, False),
    (4, 128, 0.5, False),
    (2, 64, None, True),
    (2, 256, 2.0, False),
    (1, 4096, None, False),
]


def input_build(config):
    rows, hidden, scale_ub, zero_row = config
    generator = torch.Generator(device="cpu").manual_seed(rows * 1000 + hidden)
    inp = torch.randn((rows, hidden), generator=generator)
    weight = 0.5 + torch.rand((hidden,), generator=generator)
    if zero_row:
        inp[0].zero_()
    upper = None if scale_ub is None else torch.tensor([scale_ub], dtype=torch.float32)
    return {"inp": inp, "weight": weight, "epsilon": 1e-5, "scale_ub": upper}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_rmsnorm_dynamic_fp8(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_rmsnorm_dynamic_fp8(**inputs)


def compare(ref_out, res_out, config):
    rows, hidden, scale_ub, zero_row = config
    ref_q, ref_s = ref_out
    res_q, res_s = res_out
    assert res_q.shape == ref_q.shape == (rows, hidden)
    assert res_q.dtype == ref_q.dtype == torch.float8_e4m3fn
    assert res_s.shape == ref_s.shape == (rows, 1)
    assert res_s.dtype == ref_s.dtype == torch.float32
    torch.testing.assert_close(res_s, ref_s, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(res_q.float(), ref_q.float(), rtol=0.01, atol=0.25)
    # Dynamic scale is the capped amax divided by 448 (never below the floor).
    assert (res_s >= MIN_SCALE).all()
    assert (res_s <= (FP8_MAX if scale_ub is None else scale_ub) / FP8_MAX + 1e-6).all()
    if zero_row:
        assert torch.all(res_s[0] == MIN_SCALE)


@label("v1_vllm_rmsnorm_dynamic_fp8")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_rmsnorm_dynamic_fp8(config):
    inputs = input_build(config)
    before = {k: v.clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)
    unchanged(ref_inputs, before)
    unchanged(res_inputs, before)
    assert not torch._C._is_alias_of(res_out[0], res_inputs["inp"])

    ref_time = measure(lambda: compute_ref(**ref_inputs))
    res_time = measure(lambda: compute_res(**res_inputs))
    return result(ref_time, res_time)
