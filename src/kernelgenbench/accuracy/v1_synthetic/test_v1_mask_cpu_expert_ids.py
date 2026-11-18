"""Accuracy and performance for v1_mask_cpu_expert_ids (SGLang KT EP mask).

Official entry: SGLang ``mask_cpu_expert_ids``, pinned commit
62384de1ae5e3bbd3159afd1639525b0bf39fbf9, source lines 99-116.  The upstream
function is wrapped in ``torch.compile(dynamic=True)``; the candidate binding
calls that compiled entry, so the official compiler/backend runs.

Input: ``topk_ids`` [tokens, top_k], int32 or int64, contiguous or stride-2 on
the trailing axis.  Every value >= ``num_gpu_experts`` becomes -1 in place;
the same object is returned.  The test checks the exact mask, negative
sentinels, stride preservation and object identity, then times both sides.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows202_211_test_utils import fresh, measure, result


# shape, dtype, threshold, layout.  Empty and single-row shapes cover the
# boundaries; the strided variant keeps the same values at stride 2.
COMBOS = [
    (shape, dtype, threshold, layout)
    for shape in [(0, 3), (1, 1), (7, 3), (16, 8)]
    for dtype in (torch.int32, torch.int64)
    for threshold in (0, 4, 10)
    for layout in ("contiguous", "strided")
    if layout == "contiguous" or (shape[0] > 0 and shape[1] > 1)
]


def input_build(config):
    """Build values in [-2, 11] so both sides of the threshold occur."""
    shape, dtype, threshold, layout = config
    n, k = shape
    base = (torch.arange(max(n, 1) * k * 2, dtype=dtype) % 14 - 2).reshape(max(n, 1), k * 2)
    topk_ids = base[:n, ::2] if layout == "strided" else base[:n, :k].contiguous()
    assert topk_ids.shape == shape
    assert topk_ids.is_contiguous() == (layout == "contiguous")
    return {"topk_ids": topk_ids, "num_gpu_experts": threshold}


def compute_ref(topk_ids, num_gpu_experts):
    return kernelgenbench.baseline.v1_mask_cpu_expert_ids(topk_ids, num_gpu_experts)


def compute_res(topk_ids, num_gpu_experts):
    return kernelgenbench.solution.v1_mask_cpu_expert_ids(topk_ids, num_gpu_experts)


def compare(ref_out, res_out):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert torch.equal(res_out, ref_out)


def compute_ref_performance(inputs):
    """Time the reference on independently rebuilt inputs."""
    return measure(compute_ref, lambda: fresh(inputs, kernelgenbench.device))


def compute_res_performance(inputs):
    """Time the candidate on independently rebuilt inputs."""
    return measure(compute_res, lambda: fresh(inputs, kernelgenbench.device))


@label("v1_mask_cpu_expert_ids")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_mask_cpu_expert_ids(config):
    shape, dtype, threshold, layout = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert type(side["topk_ids"]) is torch.Tensor
        assert side["topk_ids"].shape == shape
        assert side["topk_ids"].dtype == dtype
        assert side["topk_ids"].stride() == inputs["topk_ids"].stride()

    before = res_inputs["topk_ids"].clone()
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)

    # In-place contract: both sides return the exact input object.
    assert ref_out is ref_inputs["topk_ids"], "reference must return its input"
    assert res_out is res_inputs["topk_ids"], "solution must return its input"
    compare(ref_out, res_out)

    # Independent mask check, including already-negative values.
    expected = before.clone()
    expected[expected >= threshold] = -1
    assert torch.equal(res_inputs["topk_ids"], expected), "mask contract"
    assert torch.equal(
        res_inputs["topk_ids"][before < 0], before[before < 0]
    ), "negative sentinels must be unchanged"
    assert res_inputs["topk_ids"].stride() == inputs["topk_ids"].stride()

    ref_ms = compute_ref_performance(inputs)
    res_ms = compute_res_performance(inputs)
    return result(ref_ms, res_ms)
