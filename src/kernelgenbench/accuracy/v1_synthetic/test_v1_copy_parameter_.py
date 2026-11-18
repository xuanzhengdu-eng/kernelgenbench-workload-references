"""Accuracy and performance for v1_copy_parameter_ (vLLM branch).

Operator: vLLM ``copy_weight_`` (checkpoint copy into a parameter).
Official source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/utils.py#L79-L103

Contract: ``dst.copy_(src)`` in place; returns the very same destination tensor
object.  Casting follows torch copy semantics and therefore downcasts are
allowed without a policy flag.  The source is unchanged.  Strided destinations
are written through their view (only the view's elements change).

Item: v1_copy_parameter_ (723-row 637, vLLM contract).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows632_641_test_utils import (
    fresh, measure, timings)

# target dtype, source dtype, row-strided target view.
CASES = [
    (torch.float32, torch.float32, False),
    (torch.bfloat16, torch.bfloat16, False),
    (torch.float32, torch.bfloat16, False),
    (torch.bfloat16, torch.float32, False),
    (torch.int32, torch.int32, False),
    (torch.bfloat16, torch.float32, True),
]
COMBOS = [(case,) for case in CASES]


def input_build(config):
    """target: [4, 16] zeros (optionally a view with row stride 20);
    loaded_weight: [4, 16] small integers, so casts are exact.
    """
    (target_dtype, source_dtype, strided), = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    if strided:
        target = torch.zeros((4, 16 + 4), dtype=target_dtype)[:, :16]
    else:
        target = torch.zeros((4, 16), dtype=target_dtype)
    source = torch.randint(-8, 9, (4, 16), generator=generator).to(source_dtype)
    return {"target": target, "loaded_weight": source}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_copy_parameter_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_copy_parameter_(**inputs)


def compare(ref_out, res_out, ref_inputs, res_inputs):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    # vLLM returns the destination object itself, not a new tensor or None.
    assert ref_out is ref_inputs["target"], "reference must return its target"
    assert res_out is res_inputs["target"], "candidate must return its target"
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    torch.testing.assert_close(res_out.float(), ref_out.float(), rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs, kernelgenbench.device)


@label("v1_copy_parameter_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_copy_parameter_(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    source_before = inputs["loaded_weight"].clone()

    compare(compute_ref(**ref_inputs), compute_res(**res_inputs),
            ref_inputs, res_inputs)

    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["loaded_weight"].cpu(), source_before,
                                   rtol=0, atol=0)
    return timings(compute_ref_performance(**inputs),
                   compute_res_performance(**inputs))
