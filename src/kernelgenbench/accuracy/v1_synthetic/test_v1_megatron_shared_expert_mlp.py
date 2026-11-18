"""Accuracy and performance for v1_megatron_shared_expert_mlp.

Contract: the eager path of Megatron ``MLP.forward`` as used by
``SharedExpertMLP.forward``. FC1 output is split into gate/linear halves,
optionally clamped; ``silu(gate) * (linear + glu_linear_offset)`` is projected by
FC2, and an optional shared-expert gate multiplies by
``sigmoid(linear(x, gate_weight))``. The official shared expert asserts
``add_bias_linear is False``; the base MLP path (no gate) additionally covers
the optional FC1 bias.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import moe_rows222_231_checks as checks


# (sequence, batch, hidden, ffn, dtype, bias, gate, clamp, linear_offset)
COMBOS = [
    (3, 2, 8, 12, torch.float32, False, True, None, 0.0),
    (4, 1, 12, 7, torch.bfloat16, True, False, None, 0.0),
    (2, 2, 8, 6, torch.float32, True, False, 1.0, 0.5),
    (1, 1, 16, 8, torch.bfloat16, False, False, None, 0.0),
    (2, 3, 8, 6, torch.bfloat16, False, True, 1.0, 0.5),
    (5, 1, 4, 5, torch.float16, False, True, None, 0.0),
]


def input_build(config):
    sequence, batch, hidden, ffn, dtype, has_bias, has_gate, clamp, offset = config
    generator = torch.Generator(device="cpu").manual_seed(231)

    def rand(shape):
        return torch.randn(shape, dtype=dtype, generator=generator) * 0.5

    return {
        "x": rand((sequence, batch, hidden)),
        "w1": rand((2 * ffn, hidden)),
        "w2": rand((hidden, ffn)),
        "b1": rand((2 * ffn,)) if has_bias else None,
        "gate_weight": rand((1, hidden)) if has_gate else None,
        "linear_offset": offset,
        "clamp_value": clamp,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_shared_expert_mlp(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_shared_expert_mlp(**inputs)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape and res_out.dtype == ref_out.dtype == dtype
    rtol, atol = checks.tolerance(dtype)
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_megatron_shared_expert_mlp, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_megatron_shared_expert_mlp, inputs, device=kernelgenbench.device)


@label("v1_megatron_shared_expert_mlp")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_shared_expert_mlp(config):
    sequence, batch, hidden, ffn, dtype, has_bias, has_gate, clamp, offset = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    before = {
        name: value.cpu().clone()
        for name, value in res_inputs.items()
        if isinstance(value, torch.Tensor)
    }

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    assert res_out.shape == (sequence, batch, hidden)
    # The shared-expert gate is a real multiplier: removing it must change the
    # result, and so does adding the FC1 bias or enabling the clamp.
    if has_gate:
        ungated = kernelgenbench.baseline.v1_megatron_shared_expert_mlp(
            **{**res_inputs, "gate_weight": None}
        )
        gate = torch.sigmoid(
            torch.nn.functional.linear(res_inputs["x"], res_inputs["gate_weight"])
        )
        torch.testing.assert_close(
            res_out, ungated * gate,
            rtol=checks.tolerance(dtype)[0], atol=checks.tolerance(dtype)[1],
        )
        assert not torch.equal(res_out, ungated), "gate weight had no effect"
    if has_bias:
        without_bias = kernelgenbench.baseline.v1_megatron_shared_expert_mlp(
            **{**res_inputs, "b1": None}
        )
        assert not torch.equal(res_out, without_bias), "FC1 bias had no effect"
    if clamp is not None:
        unclamped = kernelgenbench.baseline.v1_megatron_shared_expert_mlp(
            **{**res_inputs, "clamp_value": None}
        )
        assert not torch.equal(res_out, unclamped), "clamp had no effect"
    for name, value in before.items():
        assert torch.equal(res_inputs[name].cpu(), value), f"{name} was modified"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
