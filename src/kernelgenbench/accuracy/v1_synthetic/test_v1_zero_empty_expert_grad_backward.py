"""Accuracy and performance for v1_zero_empty_expert_grad_backward.

DeepSpeed ``_ZeroEmptyExpertGrad.backward`` on the weight-gradient buffer of a
grouped GEMM.  ``grad`` is ``(experts, ...)`` and ``num_tokens_per_expert`` is a
1-D count per expert.

Key assertions:
* a None grad returns ``(None, None)``;
* when every expert is active the **same** grad object comes back;
* with an inactive expert the contiguous buffer is zeroed in place and returned,
  while a non-contiguous buffer is copied first, so the caller's buffer keeps its
  values and the returned copy has the inactive rows zeroed;
* the counts tensor is never modified and the dtype/shape are preserved.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows172_181_test_utils import measure_cuda
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (all_active / inactive / no_grad, strided)
COMBOS = [(mode, strided) for mode in ("active", "inactive", "none") for strided in (False, True)]


def input_build(config, dtype):
    mode, strided = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    raw = torch.randn((3, 4, 10 if strided else 5), dtype=dtype, generator=generator)
    grad = None if mode == "none" else (raw[:, :, ::2] if strided else raw)
    counts = torch.tensor([2, 0, 1] if mode == "inactive" else [1, 2, 3], dtype=torch.int64)
    return {"grad": grad, "raw": raw, "num_tokens_per_expert": counts}


def compute_ref(grad, num_tokens_per_expert):
    return kernelgenbench.baseline.v1_zero_empty_expert_grad_backward(grad, num_tokens_per_expert)


def compute_res(grad, num_tokens_per_expert):
    return kernelgenbench.solution.v1_zero_empty_expert_grad_backward(grad, num_tokens_per_expert)


def compare(ref_out, res_out, original):
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    assert ref_out[1] is None and res_out[1] is None
    if original is None:
        assert ref_out[0] is None and res_out[0] is None
        return
    assert res_out[0].shape == ref_out[0].shape
    assert res_out[0].dtype == ref_out[0].dtype
    torch.testing.assert_close(res_out[0], ref_out[0], rtol=0, atol=0)


def compute_ref_performance(grad, counts, saved):
    """Time the reference entry, restoring the gradient buffer outside the timed region."""
    def setup():
        if grad is not None:
            grad.copy_(saved)

    return measure_cuda(lambda: compute_ref(grad, counts), setup)


def compute_res_performance(grad, counts, saved):
    """Time the solution entry under the same restore protocol."""
    def setup():
        if grad is not None:
            grad.copy_(saved)

    return measure_cuda(lambda: compute_res(grad, counts), setup)


@label("v1_zero_empty_expert_grad_backward")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("config", COMBOS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_accuracy_v1_zero_empty_expert_grad_backward(config, dtype):
    mode, strided = config
    cpu_inputs = input_build(config, dtype)
    counts = cpu_inputs["num_tokens_per_expert"]

    def cuda_side():
        # Build the slicing on the GPU: .cuda() of a gapped slice would densify it,
        # and the non-contiguous branch must see a genuinely strided buffer.
        base = cpu_inputs["raw"].cuda()
        moved_counts = counts.cuda()
        if cpu_inputs["grad"] is None:
            return None, moved_counts
        return (base[:, :, ::2] if strided else base), moved_counts

    ref_grad, ref_counts = cuda_side()
    res_grad, res_counts = cuda_side()
    ref_ptr = None if ref_grad is None else ref_grad.data_ptr()
    res_ptr = None if res_grad is None else res_grad.data_ptr()

    ref_out = compute_ref(ref_grad, ref_counts)
    res_out = compute_res(res_grad, res_counts)
    compare(ref_out, res_out, ref_grad)

    inactive = cpu_inputs["num_tokens_per_expert"] == 0
    has_inactive = bool(inactive.any()) and mode != "none"
    expected_original = cpu_inputs["grad"]
    expected_output = None if expected_original is None else expected_original.clone()
    if has_inactive:
        expected_output[inactive] = 0

    for side, grad, out, ptr, counts_now in (
            ("reference", ref_grad, ref_out, ref_ptr, ref_counts),
            ("solution", res_grad, res_out, res_ptr, res_counts)):
        assert torch.equal(counts_now.cpu(), counts), f"{side} modified the counts"
        if grad is None:
            assert out[0] is None
            continue
        if has_inactive and not strided:
            # contiguous + inactive: zeroed in place, same object
            assert out[0] is grad, f"{side} did not return the in-place buffer"
            assert grad.data_ptr() == ptr
            torch.testing.assert_close(grad.cpu(), expected_output, rtol=0, atol=0,
                                       msg=f"{side} in-place buffer")
        elif has_inactive and strided:
            # non-contiguous + inactive: copy returned, caller buffer untouched
            assert out[0] is not grad, f"{side} wrote through a non-contiguous view"
            assert out[0].is_contiguous()
            torch.testing.assert_close(out[0].cpu(), expected_output, rtol=0, atol=0,
                                       msg=f"{side} masked copy")
            torch.testing.assert_close(grad.cpu(), expected_original, rtol=0, atol=0,
                                       msg=f"{side} caller buffer")
        else:
            # every expert active: the very same object, untouched
            assert out[0] is grad, f"{side} copied an active-only gradient"
            torch.testing.assert_close(grad.cpu(), expected_original, rtol=0, atol=0,
                                       msg=f"{side} active gradient")
    if has_inactive and not strided:
        # both sides saw identical in-place zeroing
        torch.testing.assert_close(ref_grad.cpu(), res_grad.cpu(), rtol=0, atol=0)

    saved_grad = None if cpu_inputs["grad"] is None else cpu_inputs["grad"].cuda()

    ref_time = compute_ref_performance(ref_grad, ref_counts, saved_grad)
    res_time = compute_res_performance(res_grad, res_counts, saved_grad)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
