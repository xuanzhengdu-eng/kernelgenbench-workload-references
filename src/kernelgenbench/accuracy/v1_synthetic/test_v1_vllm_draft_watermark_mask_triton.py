"""Accuracy and performance for v1_vllm_draft_watermark_mask_ (CUDA Triton branch).

Official source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/watermark.py#L407-L441

Contract: return a bool ``[R]`` mask that is true exactly when the request is
enabled and its current context was not seen in the allowed history.  History
is ``prior_contexts[row, step - max_history:step]`` or the full prefix when
``max_history`` is ``None``; the repeated-context body also looks at the prompt
tail/history when enabled.  ``prior_contexts[row, step]`` is updated in place
with ``contexts[row]``.  On CUDA the pinned Triton kernel exits before recording a
padded row (req_index < 0), so that row keeps its prior history and returns
False; this is the branch tested here.

Item: v1_vllm_draft_watermark_mask_ (723-row 591, CUDA Triton backend).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows582_591_checks as checks


# max_history, include_prompt, context pattern, enabled pattern.
CASES = [
    (None, False, "repeated", "mixed"),
    (2, True, "repeated", "mixed"),
    (1, False, "fresh", "all"),
    (3, True, "fresh", "none"),
]
COMBOS = [(case, dtype) for case in CASES for dtype in (torch.int32, torch.int64)]

# Independently derived from the four input patterns, CPU branch.
EXPECTED = {
    (None, False, "repeated", "mixed"): [False, False, False, False, False],
    (2, True, "repeated", "mixed"): [False, False, False, False, True],
    (1, False, "fresh", "all"): [True, False, True, False, True],
    (3, True, "fresh", "none"): [False, False, False, False, False],
}


def input_build(config):
    """all_token_ids int32/int64 [2, 7], prior_contexts [6, 4, 3], contexts [5, 3]."""
    (hist, inc, mode, enabled_mode), dtype = config
    all_ids = torch.tensor([[1, 2, 1, 2, 1, 3, 0], [7, 8, 9, 9, 8, 9, 0]], dtype=dtype)
    req = torch.tensor([0, 0, 1, -1, 1])
    contexts = torch.tensor([[1, 2, 1], [1, 2, 1], [7, 8, 9], [-1, -1, -1], [8, 9, 9]],
                            dtype=dtype)
    prior = torch.full((6, 4, 3), -9, dtype=dtype)
    steps = torch.tensor([1, 0, 2, 1, 3])
    if mode == "repeated":
        prior[0, 0] = contexts[0]
        prior[2, 1] = contexts[2]
        prior[4, 0] = contexts[4]
    enabled = torch.tensor(
        [True, False, True, True, True] if enabled_mode == "mixed"
        else [enabled_mode == "all"] * 5, dtype=torch.bool)
    return {
        "all_token_ids": all_ids,
        "req_indices": req,
        "prompt_lens": torch.tensor([2, 1]),
        "total_lens": torch.tensor([6, 6]),
        "prior_contexts": prior,
        "contexts": contexts,
        "steps": steps,
        "enabled": enabled,
        "max_history": hist,
        "include_prompt": inc,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_draft_watermark_mask_triton(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_draft_watermark_mask_triton(**inputs)


def compare(ref_out, res_out):
    """Bool ``[R]`` mask: same shape/dtype and identical values."""
    assert isinstance(res_out, torch.Tensor)
    assert res_out.dtype == ref_out.dtype == torch.bool
    assert res_out.shape == ref_out.shape
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return checks.measure(compute_ref, inputs, inputs["contexts"].device)


def compute_res_performance(**inputs):
    return checks.measure(compute_res, inputs, inputs["contexts"].device)


@label("v1_vllm_draft_watermark_mask_triton")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_draft_watermark_mask_triton(config):
    case, dtype = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    contexts_before = ref_inputs["contexts"].clone()
    other_before = {key: value.clone() for key, value in res_inputs.items()
                    if isinstance(value, torch.Tensor)
                    and key not in ("prior_contexts", "contexts")}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)
    assert res_out.shape == (5,)

    expected = torch.tensor(EXPECTED[case], dtype=torch.bool)
    torch.testing.assert_close(res_out.cpu(), expected, rtol=0, atol=0)
    # prior_contexts is written for active rows; the padded row keeps -9.
    torch.testing.assert_close(res_inputs["prior_contexts"], ref_inputs["prior_contexts"],
                               rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["contexts"], contexts_before, rtol=0, atol=0)
    assert torch.all(res_inputs["prior_contexts"][3] == -9)
    for key, value in other_before.items():
        torch.testing.assert_close(res_inputs[key], value, rtol=0, atol=0)

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return checks.result(ref_time, res_time)
