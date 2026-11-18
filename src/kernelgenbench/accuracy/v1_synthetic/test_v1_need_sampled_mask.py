"""Accuracy and performance for v1_need_sampled_mask.

Official source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/pp_utils.py#L38-L52

Contract: NumPy metadata arrays of shape ``[B]``.  A request needs its sampled
token when ``num_computed_tokens + num_scheduled_tokens >= prefill_len``.  If
``max_seq_len`` is not ``None``, a final prefill chunk whose next token reaches
the cap is excluded while ``is_prefilling`` is true.  The result is a bool
``[B]`` array, or ``None`` when no request needs a sample.  Inputs are
read-only.  This is CPU metadata, so timing is wall-clock, not GPU kernel time.

Item: v1_need_sampled_mask (723-row 589).
"""
import numpy as np
import pytest

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows582_591_checks as checks


# (old_computed, scheduled, prefill_len, max_seq_len, is_prefilling); use_cap
# decides whether max_seq_len is supplied or None.
CASES = [
    ((0, 5, 10), (4, 3, 1), (8, 8, 10), (100, 9, 10), (True, True, False)),
    ((0, 0), (1, 2), (6, 6), (10, 10), (True, True)),
    ((), (), (), (), ()),
]
COMBOS = [(case, use_cap) for case in CASES for use_cap in (False, True)]


def input_build(config):
    """Five ``[B]`` arrays: int64 counts and a bool prefilling flag."""
    (old, scheduled, prefill, cap, flags), use_cap = config
    return {
        "old_computed": np.asarray(old, dtype=np.int64),
        "scheduled": np.asarray(scheduled, dtype=np.int64),
        "prefill_len": np.asarray(prefill, dtype=np.int64),
        "max_seq_len": np.asarray(cap, dtype=np.int64) if use_cap else None,
        "is_prefilling": np.asarray(flags, dtype=np.bool_),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_need_sampled_mask(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_need_sampled_mask(**inputs)


def compare(ref_out, res_out):
    assert (ref_out is None) == (res_out is None)
    if ref_out is not None:
        assert isinstance(res_out, np.ndarray)
        assert res_out.dtype == np.bool_
        assert res_out.shape == ref_out.shape
        assert np.array_equal(res_out, ref_out)


def compute_ref_performance(**inputs):
    return checks.measure(compute_ref, inputs, "cpu")


def compute_res_performance(**inputs):
    return checks.measure(compute_res, inputs, "cpu")


@label("v1_need_sampled_mask")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_need_sampled_mask(config):
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, "cpu")
    res_inputs = checks.fresh(inputs, "cpu")
    before = {key: value.copy() for key, value in res_inputs.items()
              if isinstance(value, np.ndarray)}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    for key, value in before.items():
        assert np.array_equal(res_inputs[key], value)
        assert np.array_equal(ref_inputs[key], value)

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return checks.result(ref_time, res_time)
