"""Accuracy and performance for v1_vllm_min_token_mask_.

Official source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/logits_processor/builtin.py#L254-L288

Contract: with min-token filtering active, every indexed stop-token logit is
set to ``-inf`` in place and the same logits object is returned.  If
``restore_rows`` is non-empty, entries are restored from a pre-mask copy when
the whole row became ``-inf`` and the saved value was finite.  Restore rows are
grouped consecutively by the source.  The pinned implementation builds
``neg_inf_tensor`` as float32, so the entry only accepts float32 logits; fp16
and bf16 logits raise ``RuntimeError`` from ``index_put_`` and are therefore not
part of ``COMBOS``.

Item: v1_vllm_min_token_mask_ (723-row 582).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows582_591_checks as checks


# Axes: the four source branches (disabled, mask only, restore, already all
# -inf) plus a finite-other-token row where a saved stop token must stay
# masked, because the row did not become entirely -inf.
CASES = ("disabled", "mask", "restore", "already_inf", "partial_row")
COMBOS = [(case, torch.float32) for case in CASES]


def _logits(dtype):
    return torch.tensor(
        [[1., 2., 3., 4.],
         [5., float('-inf'), float('-inf'), float('-inf')],
         [0., 1., 2., 3.]],
        dtype=dtype,
    )


def input_build(config):
    """logits [3,4]; rows/tokens int32 [3] or [0]/[1]; enabled is a Python bool.

    ``restore``/``already_inf`` restore row 1 token 0.  ``partial_row`` also
    restores row 1 token 0 but first makes token 3 finite, so the row is not
    entirely masked and the saved value must not come back.  ``disabled`` keeps
    an empty restore slice and must leave the logits untouched.
    """
    case, dtype = config
    logits = _logits(dtype)
    if case == "already_inf":
        logits[1, 0] = -torch.inf
    if case == "partial_row":
        logits[1, 3] = 6.
    rows = torch.tensor([0, 1, 2], dtype=torch.int32)
    tokens = torch.tensor([1, 0, 2], dtype=torch.int32)
    needs_restore = case in ("restore", "already_inf", "partial_row")
    restore_rows = torch.tensor([1] if needs_restore else [], dtype=torch.int32)
    restore_tokens = torch.tensor([0] if needs_restore else [], dtype=torch.int32)
    return {
        "logits": logits,
        "mask_rows": rows,
        "mask_tokens": tokens,
        "restore_rows": restore_rows,
        "restore_tokens": restore_tokens,
        "enabled": case != "disabled",
    }


def compute_ref(logits, mask_rows, mask_tokens, restore_rows, restore_tokens, enabled):
    return kernelgenbench.baseline.v1_vllm_min_token_mask_(
        logits, mask_rows, mask_tokens, restore_rows, restore_tokens, enabled
    )


def compute_res(logits, mask_rows, mask_tokens, restore_rows, restore_tokens, enabled):
    """Candidate side: the verifier-bound entry with the reference signature."""
    return kernelgenbench.solution.v1_vllm_min_token_mask_(
        logits, mask_rows, mask_tokens, restore_rows, restore_tokens, enabled
    )


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out is not None
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0, equal_nan=True)


def compute_ref_performance(logits, mask_rows, mask_tokens, restore_rows, restore_tokens, enabled):
    initial = logits.clone()

    def call():
        logits.copy_(initial)
        return compute_ref(logits, mask_rows, mask_tokens, restore_rows,
                           restore_tokens, enabled)

    return checks.measure(call, {}, logits.device)


def compute_res_performance(logits, mask_rows, mask_tokens, restore_rows, restore_tokens, enabled):
    initial = logits.clone()

    def call():
        logits.copy_(initial)
        return compute_res(logits, mask_rows, mask_tokens, restore_rows,
                           restore_tokens, enabled)

    return checks.measure(call, {}, logits.device)


@label("v1_vllm_min_token_mask_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_min_token_mask_(config):
    case, dtype = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    snapshot = ref_inputs["logits"].clone()
    index_inputs = {key: value.clone() for key, value in ref_inputs.items()
                    if isinstance(value, torch.Tensor) and key != "logits"}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # Both entries mutate and return the caller's tensor.
    assert ref_out is ref_inputs["logits"]
    assert res_out is res_inputs["logits"]
    # Auxiliary index tensors are read-only.
    for key, value in index_inputs.items():
        torch.testing.assert_close(ref_inputs[key], value, rtol=0, atol=0)
        torch.testing.assert_close(res_inputs[key], value, rtol=0, atol=0)

    if case == "disabled":
        torch.testing.assert_close(res_inputs["logits"], snapshot, rtol=0, atol=0)
    else:
        # Masked positions are exactly the (row, token) pairs.
        expected = snapshot.clone()
        expected.index_put_((inputs["mask_rows"].long(), inputs["mask_tokens"].long()),
                            torch.tensor(float("-inf"), dtype=dtype))
        if case == "restore":
            expected[1, 0] = 5.
        if case == "already_inf":
            expected[1, 0] = -torch.inf
        if case == "partial_row":
            expected[1, 0] = -torch.inf
        torch.testing.assert_close(res_inputs["logits"], expected, rtol=0, atol=0)
        torch.testing.assert_close(ref_inputs["logits"], expected, rtol=0, atol=0)

    if case == "restore":
        assert res_out[1, 0].item() == 5.
    if case in ("already_inf", "partial_row"):
        assert torch.isneginf(res_out[1, 0])
    if case == "partial_row":
        assert res_out[1, 3].item() == 6.

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return checks.result(ref_time, res_time)
