"""Accuracy and performance for v1_moe_preapply_weights.

Contract: SGLang ``apply_topk_weights_cpu(need_apply, topk_weights, inputs)``
(fixed-source argument order). Disabled returns the identical two input objects;
enabled casts the weights to the input dtype first, multiplies elementwise and
returns fresh values plus an FP32 ones tensor of the weights' shape. Nothing is
mutated in either branch.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import moe_rows222_231_checks as checks


# (rows, hidden, dtype, need_apply)
COMBOS = [
    (n, h, dtype, need_apply)
    for n, h in [(0, 8), (1, 7), (3, 16), (7, 33)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for need_apply in (False, True)
]


def input_build(config):
    rows, hidden, dtype, need_apply = config
    generator = torch.Generator(device="cpu").manual_seed(230)
    inputs = torch.randn((rows, hidden), generator=generator).to(dtype)
    topk_weights = torch.rand((rows, 1), generator=generator, dtype=torch.float32)
    return {"inputs": inputs, "topk_weights": topk_weights, "need_apply": need_apply}


def compute_ref(need_apply, topk_weights, inputs):
    return kernelgenbench.baseline.v1_moe_preapply_weights(need_apply, topk_weights, inputs)


def compute_res(need_apply, topk_weights, inputs):
    return kernelgenbench.solution.v1_moe_preapply_weights(need_apply, topk_weights, inputs)


def compare(ref_out, res_out, dtype=None, need_apply=False):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 2
    assert ref_out[0].shape == res_out[0].shape and res_out[0].dtype == dtype
    assert ref_out[1].dtype == res_out[1].dtype == torch.float32
    torch.testing.assert_close(res_out[0], ref_out[0], rtol=0, atol=0)
    torch.testing.assert_close(res_out[1], ref_out[1], rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_moe_preapply_weights, inputs)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_moe_preapply_weights, inputs)


@label("v1_moe_preapply_weights")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_moe_preapply_weights(config):
    rows, hidden, dtype, need_apply = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, "cpu")
    res_inputs = checks.fresh(inputs, "cpu")
    before = {
        name: value.clone()
        for name, value in res_inputs.items()
        if isinstance(value, torch.Tensor)
    }

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype, need_apply)

    if not need_apply:
        # Disabled branch returns the very same objects, not copies.
        assert res_out[0] is res_inputs["inputs"]
        assert res_out[1] is res_inputs["topk_weights"]
    else:
        # Enabled branch rounds the weights to the input dtype before multiplying
        # and replaces the weights with FP32 ones; both results are fresh.
        expected = res_inputs["inputs"] * res_inputs["topk_weights"].to(dtype)
        torch.testing.assert_close(res_out[0], expected, rtol=0, atol=0)
        assert res_out[0].dtype == dtype
        assert res_out[1].dtype == torch.float32
        assert bool((res_out[1] == 1).all())
        if res_out[0].numel() and res_inputs["inputs"].numel():
            assert res_out[0].data_ptr() != res_inputs["inputs"].data_ptr()
        if res_out[1].numel() and res_inputs["topk_weights"].numel():
            assert res_out[1].data_ptr() != res_inputs["topk_weights"].data_ptr()
    for name, value in before.items():
        assert torch.equal(res_inputs[name], value), f"{name} was modified"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
