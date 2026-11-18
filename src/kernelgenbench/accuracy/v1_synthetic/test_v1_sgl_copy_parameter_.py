"""Accuracy and performance for v1_sgl_copy_parameter_ (SGLang branch).

Operator: SGLang ``copy_with_check`` (shape-checked parameter copy with a
downcast policy).
Official source (pinned commit 62384de1ae5e3bbd3159afd1639525b0bf39fbf9):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/parameter.py#L51-L79

Contract: asserts equal shapes, copies in place and returns None.  Equal dtypes
copy directly.  Different dtypes are ranked (fp8=0, fp16/bf16=1, fp32=2,
fp64=3): an unknown dtype raises ``ValueError('Unsupported copy between
dtypes: ...')``; copying into a lower rank raises ``ValueError('Downcasting not
allowed: ...')`` unless ``SGLANG_QUANT_ALLOW_DOWNCASTING`` is enabled.  On error
the destination is untouched.  Returning None and refusing downcasts is what
separates this item from vLLM's ``copy_weight_``.

Item: v1_sgl_copy_parameter_ (723-row 637, SGLang contract).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows632_641_test_utils import (
    fresh, measure, timings)

# target dtype, source dtype, allow_downcast, row-strided target, expected error prefix.
CASES = [
    (torch.float32, torch.float32, False, False, None),
    (torch.bfloat16, torch.bfloat16, False, False, None),
    (torch.float32, torch.bfloat16, False, False, None),
    (torch.bfloat16, torch.float32, True, False, None),
    (torch.bfloat16, torch.float32, False, False, "Downcasting not allowed"),
    (torch.float32, torch.int32, False, False, "Unsupported copy between dtypes"),
    (torch.int32, torch.int32, False, False, None),
    (torch.bfloat16, torch.float32, True, True, None),
    (torch.float32, torch.float32, False, True, None),
]
COMBOS = [(case,) for case in CASES]


def input_build(config):
    """target: [4, 16] zeros (optionally a view with row stride 20);
    loaded_weight: [4, 16] small integers (exact in fp32/bf16/int32).
    """
    (target_dtype, source_dtype, allow_downcast, strided, _expected), = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    if strided:
        target = torch.zeros((4, 16 + 4), dtype=target_dtype)[:, :16]
    else:
        target = torch.zeros((4, 16), dtype=target_dtype)
    source = torch.randint(-8, 9, (4, 16), generator=generator).to(source_dtype)
    return {"target": target, "loaded_weight": source,
            "allow_downcast": allow_downcast}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_copy_parameter_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_copy_parameter_(**inputs)


def observe(fn, inputs):
    try:
        return fn(**inputs), None
    except Exception as exc:  # noqa: BLE001 - the contract is the exception type/message
        return None, exc


def compare(ref_observed, res_observed, expected, ref_inputs, res_inputs):
    (ref_out, ref_error), (res_out, res_error) = ref_observed, res_observed
    if expected is None:
        assert ref_error is None, f"reference unexpectedly raised {ref_error}"
        assert res_error is None, f"candidate raised {res_error}"
        assert ref_out is None and res_out is None, "copy_with_check must return None"
        torch.testing.assert_close(res_inputs["target"].float(),
                                   ref_inputs["target"].float(), rtol=0, atol=0)
    else:
        assert isinstance(ref_error, ValueError), "reference must raise ValueError"
        assert isinstance(res_error, ValueError), \
            f"candidate must raise ValueError, got {res_error!r}"
        assert str(ref_error).startswith(expected), str(ref_error)
        assert str(res_error).startswith(expected), str(res_error)
        # A rejected copy leaves the destination at its initial value.
        assert torch.count_nonzero(res_inputs["target"]) == 0


def compute_ref_performance(**inputs):
    return measure(lambda **kw: observe(compute_ref, kw), inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return measure(lambda **kw: observe(compute_res, kw), inputs, kernelgenbench.device)


@label("v1_sgl_copy_parameter_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_copy_parameter_(config):
    expected = config[0][4]
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    source_before = inputs["loaded_weight"].clone()

    compare(observe(compute_ref, ref_inputs), observe(compute_res, res_inputs),
            expected, ref_inputs, res_inputs)

    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["loaded_weight"].cpu(), source_before,
                                   rtol=0, atol=0)
    return timings(compute_ref_performance(**inputs),
                   compute_res_performance(**inputs))
