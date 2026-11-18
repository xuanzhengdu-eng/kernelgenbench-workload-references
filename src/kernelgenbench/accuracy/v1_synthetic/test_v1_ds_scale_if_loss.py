"""Accuracy and performance for v1_ds_scale_if_loss (row 161).

Official entry: DeepSpeed ``ZeROOptimizer.scale_if_loss`` at commit
8af59b2e7a91c227081617240e98aa282eadc5f9,
deepspeed/runtime/base_optimizer.py#L428-L441, with ``maybe_loss_for_backward``
from deepspeed/runtime/utils.py#L1332-L1343 and ``LossScalerBase.scale_loss``
from deepspeed/runtime/fp16/loss_scaler.py#L165-L169.  The test checks that
only one-element tensors with a grad_fn are scaled, that the custom and
internal branches agree numerically, and that every passthrough preserves
object identity.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows_152_161_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# 'loss' has a grad_fn and one element; 'leaf' is a one-element leaf;
# 'vector' has several elements; 'number' is a Python float.
CASES = ["loss", "leaf", "vector", "number"]
COMBOS = [
    (case, dtype, mode)
    for case in CASES
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for mode in ("custom", "internal", "none")
]


def input_build(config):
    case, dtype, mode = config
    if case == "number":
        value = 3.5
    else:
        value = torch.arange(1, 3 if case == "vector" else 2, dtype=dtype, requires_grad=True)
        if case != "leaf":
            value = value * 1.5
    return {"value": value, "scale": 8.0, "mode": mode}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_scale_if_loss(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_scale_if_loss(**inputs)


def compare(ref_out, res_out, ref_value, res_value, mode, case):
    is_loss = case == "loss"
    if isinstance(ref_value, torch.Tensor):
        if is_loss and mode in ("custom", "internal"):
            # A loss tensor is scaled into a new tensor; the input object stays
            # untouched and is still the caller's tensor.
            for out, value in ((ref_out, ref_value), (res_out, res_value)):
                assert isinstance(out, torch.Tensor)
                assert out.shape == value.shape and out.dtype == value.dtype
                assert out.device == value.device
                assert out is not value
            torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)
            assert torch.equal(ref_out.detach() * 1, ref_value.detach() * 8.0)
        else:
            assert ref_out is ref_value, "non-loss value must pass through by identity"
            assert res_out is res_value, "non-loss value must pass through by identity"
    else:
        assert ref_out is ref_value and res_out is res_value
        assert isinstance(ref_out, float) and isinstance(res_out, float)


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure(compute_res, inputs)


@label("v1_ds_scale_if_loss")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_scale_if_loss(config):
    case, dtype, mode = config
    inputs = input_build(config)
    value = inputs["value"]
    saved = value.detach().clone() if isinstance(value, torch.Tensor) else value

    ref_inputs = fresh(inputs, kernelgenbench.device)
    if isinstance(saved, torch.Tensor):
        saved = saved.to(kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs["value"], res_inputs["value"], mode, case)

    if isinstance(saved, torch.Tensor):
        assert torch.equal(ref_inputs["value"].detach(), saved)
        assert torch.equal(res_inputs["value"].detach(), saved)
    else:
        assert ref_inputs["value"] == res_inputs["value"] == saved

    ref_time = compute_ref_performance(ref_inputs)
    res_time = compute_res_performance(res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
