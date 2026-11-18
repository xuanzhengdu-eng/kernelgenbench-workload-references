"""Accuracy and performance for v1_softmax_with_sink (NVIDIA Megatron).

Item: Megatron SoftmaxOne.forward (row 81).
Official source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_softmax.py#L154-L176

The sink logit participates in the softmax denominator but is removed from the
returned tensor, so the output does NOT sum to one over the K returned entries:
``sum(out) = 1 - p_sink < 1``. Output dtype follows torch.cat promotion of
``x`` and ``sink`` (a float32 sink with fp16 x yields float32).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows72_81_checks as checks

NAME = "v1_softmax_with_sink"

# (batch, heads, queries, keys, dtype, float_sink)
COMBOS = [
    (2, 3, 4, 9, torch.float32, False),
    (1, 4, 16, 32, torch.float32, True),
    (2, 3, 4, 9, torch.float16, False),
    (2, 3, 4, 9, torch.float16, True),
    (2, 3, 4, 9, torch.bfloat16, False),
    (2, 3, 4, 9, torch.bfloat16, True),
    (1, 2, 1, 1, torch.float32, False),
]
TOLERANCE = {
    torch.float32: (1e-5, 1e-6),
    torch.float16: (2e-2, 2e-2),
    torch.bfloat16: (3e-2, 3e-2),
}


def input_build(config):
    """x=(B,H,Q,K); sink=(H,) promoted dtype when float_sink is requested."""
    batch, heads, queries, keys, dtype, float_sink = config
    gen = torch.Generator().manual_seed(20261109)
    return {
        "x": torch.randn((batch, heads, queries, keys), dtype=dtype, generator=gen),
        "sink": torch.randn((heads,), dtype=torch.float32 if float_sink else dtype,
                            generator=gen),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_softmax_with_sink(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_softmax_with_sink(**inputs)


def compare(ref_out, res_out, ref_inputs, config):
    batch, heads, queries, keys, dtype, float_sink = config
    sink_dtype = torch.float32 if float_sink else dtype
    src = torch.cat((ref_inputs["x"],
                     ref_inputs["sink"].reshape(1, -1, 1, 1).expand(
                         batch, -1, queries, -1)), dim=-1)
    assert res_out.shape == ref_out.shape == (batch, heads, queries, keys)
    assert res_out.dtype == ref_out.dtype == src.dtype
    promoted = src.dtype
    rtol, atol = TOLERANCE[promoted]
    torch.testing.assert_close(res_out.float(), ref_out.float(),
                               rtol=rtol, atol=atol)
    # Independent formula, including the sink in the denominator.
    expected = torch.softmax(src, dim=-1)[..., :-1]
    torch.testing.assert_close(ref_out.float(), expected.float(),
                               rtol=rtol, atol=atol)
    # The sink output was removed: the returned probabilities sum below one.
    totals = res_out.float().sum(dim=-1)
    assert bool((totals < 1.0).all()), "sink must remain in the denominator"
    assert bool((totals > 0.0).all())
    assert not torch._C._is_alias_of(res_out, ref_inputs["x"])
    assert not torch._C._is_alias_of(res_out, ref_inputs["sink"])


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_softmax_with_sink,
                          inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_softmax_with_sink,
                          inputs, kernelgenbench.device)


@label(NAME)
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_softmax_with_sink(config):
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    batch, heads, queries, keys, dtype, float_sink = config
    assert ref_inputs["x"].shape == (batch, heads, queries, keys)
    assert ref_inputs["sink"].shape == (heads,)
    before = {k: v.clone() for k, v in ref_inputs.items() if isinstance(v, torch.Tensor)}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, config)

    for name, value in before.items():
        assert torch.equal(value, ref_inputs[name]), f"reference modified {name}"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
