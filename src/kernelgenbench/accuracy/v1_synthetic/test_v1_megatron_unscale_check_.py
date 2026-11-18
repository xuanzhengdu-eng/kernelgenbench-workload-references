"""Accuracy and performance for v1_megatron_unscale_check_.

Megatron's fused unscale/nonfinite check on a single-rank NCCL group:
``torch._amp_foreach_non_finite_check_and_unscale_`` scales every main gradient
by ``inv_scale`` and raises ``found_inf`` on NaN/Inf; ``found_inf`` is reset to 0
first and MAX-reduced afterwards.  A stub optimizer skips the fused call but
still resets and reduces.  The return value is the Python bool
``found_inf.item() > 0``.

Multirank note: two NCCL ranks cannot share this assignment's single visible
H800 ("Duplicate GPU detected"), so the real all_reduce runs with world size one.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows172_181_test_utils import fresh_tensor, measure_cuda, nccl_world
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (nonfinite, nan_instead_of_inf, stub, dtype)
# fp16/fp32 only: the official fused CUDA op raises NotImplementedError for
# BFloat16 in this torch build, so bf16 is an upstream-unsupported dtype here.
COMBOS = [
    (False, False, False, torch.float32),
    (False, False, False, torch.float16),
    (True, False, False, torch.float32),
    (True, True, False, torch.float16),
    (True, False, True, torch.float32),
]


def input_build(config):
    nonfinite, nan, stub, dtype = config
    grads = [torch.arange(6, dtype=torch.float32).reshape(2, 3) + 1 + i for i in range(2)]
    if nonfinite:
        grads[0] = grads[0].clone()
        grads[0][0, 1] = float("nan") if nan else float("inf")
    return {"grads": [g.to(dtype) for g in grads],
            "found_inf": torch.tensor([7.0], dtype=torch.float32),
            "inv_scale": torch.tensor([0.25], dtype=torch.float32),
            "is_stub_optimizer": stub}


def compute_ref(grads, found_inf, inv_scale, group, is_stub_optimizer):
    return kernelgenbench.baseline.v1_megatron_unscale_check_(
        grads, found_inf, inv_scale, group, is_stub_optimizer)


def compute_res(grads, found_inf, inv_scale, group, is_stub_optimizer):
    return kernelgenbench.solution.v1_megatron_unscale_check_(
        grads, found_inf, inv_scale, group, is_stub_optimizer)


def compare(ref_out, res_out, expected_flag, ref_found, res_found):
    """Both sides return the same Python bool and leave the same reduced found_inf."""
    assert type(ref_out) is bool and type(res_out) is bool
    assert ref_out == res_out == expected_flag, (ref_out, res_out, expected_flag)
    assert torch.equal(ref_found.cpu(), res_found.cpu())
    assert ref_found.item() == (1.0 if expected_flag else 0.0)


def compute_ref_performance(grads, found_inf, inv_scale, group, is_stub_optimizer, saved, reset):
    """Time the reference entry, restoring grads and found_inf outside the timed region."""
    def setup():
        for grad, original in zip(grads, saved):
            grad.copy_(original)
        found_inf.fill_(reset)

    return measure_cuda(
        lambda: compute_ref(grads, found_inf, inv_scale, group, is_stub_optimizer), setup)


def compute_res_performance(grads, found_inf, inv_scale, group, is_stub_optimizer, saved, reset):
    """Time the solution entry under the same restore protocol."""
    def setup():
        for grad, original in zip(grads, saved):
            grad.copy_(original)
        found_inf.fill_(reset)

    return measure_cuda(
        lambda: compute_res(grads, found_inf, inv_scale, group, is_stub_optimizer), setup)


@label("v1_megatron_unscale_check_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_unscale_check_(config):
    nonfinite, nan, stub, dtype = config
    group = nccl_world()
    cpu_inputs = input_build(config)

    ref_grads = [fresh_tensor(g) for g in cpu_inputs["grads"]]
    res_grads = [fresh_tensor(g) for g in cpu_inputs["grads"]]
    ref_found = fresh_tensor(cpu_inputs["found_inf"])
    res_found = fresh_tensor(cpu_inputs["found_inf"])
    ref_scale = fresh_tensor(cpu_inputs["inv_scale"])
    res_scale = fresh_tensor(cpu_inputs["inv_scale"])

    ref_out = compute_ref(ref_grads, ref_found, ref_scale, group, stub)
    res_out = compute_res(res_grads, res_found, res_scale, group, stub)

    expected_flag = bool(nonfinite and not stub)
    # found_inf is reset from 7 and reduced to 0 or 1 on both sides.
    compare(ref_out, res_out, expected_flag, ref_found, res_found)

    for side, grads, found in (("reference", ref_grads, ref_found), ("solution", res_grads, res_found)):
        for i, (grad, original) in enumerate(zip(grads, cpu_inputs["grads"])):
            assert grad.dtype == dtype and grad.shape == original.shape
            if stub:
                torch.testing.assert_close(grad.cpu(), original, rtol=0, atol=0,
                                           msg=f"{side} stub scaled gradient {i}")
            else:
                expected = (original.float() * 0.25).to(dtype)
                torch.testing.assert_close(grad.cpu(), expected, rtol=0, atol=0, equal_nan=True,
                                           msg=f"{side} unscaled gradient {i}")
        if not nonfinite:
            assert found.item() == 0.0

    ref_time = compute_ref_performance(ref_grads, ref_found, ref_scale, group, stub,
                                       cpu_inputs["grads"], 7.0)
    res_time = compute_res_performance(res_grads, res_found, res_scale, group, stub,
                                       cpu_inputs["grads"], 7.0)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
