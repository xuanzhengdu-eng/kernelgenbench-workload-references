"""Accuracy and performance for v1_megatron_clip_grad_.

Megatron ``clip_grad_by_total_norm_fp32``: a parameter list whose ``.grad``
(fp32 CUDA) or ``.decoupled_grad`` (fp32/bf16) buffers are scaled in place by
``max_norm / (total_norm + 1e-6)`` when that coefficient is below one, and left
bit-identical otherwise.  The function returns None.

The legacy reference mixed in a DeepSpeed tensor-list branch; that branch is the
official entry of CSV row 175 ``v1_clip_tensors_global_norm_`` and is
deduplicated there.  Parameters without a gradient are skipped.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows172_181_test_utils import clone_like, fresh_tensor, measure_cuda
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (max_norm, total_norm, use_decoupled_grad, dtype, layout, missing_grad)
COMBOS = [
    (1.0, 2.0, False, torch.float32, "contiguous", False),
    (1.0, 0.5, False, torch.float32, "contiguous", False),
    (2.0, 0.0, False, torch.float32, "contiguous", False),
    (1.0, 2.0, False, torch.float32, "strided", False),
    (1.0, 2.0, False, torch.float32, "contiguous", True),
    (1.0, 2.0, True, torch.float32, "contiguous", False),
    (1.0, 2.0, True, torch.bfloat16, "strided", False),
    (1.0, 0.25, True, torch.bfloat16, "contiguous", False),
]


def input_build(config):
    max_norm, total_norm, decoupled, dtype, layout, missing = config
    params = []
    for i in range(3):
        param = torch.nn.Parameter(torch.zeros((3, 4), dtype=torch.float32))
        if missing and i == 2:
            params.append(param)
            continue
        # 3x4 gradient per parameter; the strided case is a column slice of a
        # (3, 8) buffer, so the gradient keeps shape (3, 4) with stride (8, 2).
        if layout == "strided":
            grad = (torch.arange(24, dtype=torch.float32).reshape(3, 8) * (i + 1)).to(dtype)[:, ::2]
        else:
            grad = (torch.arange(12, dtype=torch.float32).reshape(3, 4) * (i + 1)).to(dtype)
        if decoupled:
            param.decoupled_grad = grad
        else:
            param.grad = grad
        params.append(param)
    return {"parameters": params, "max_norm": max_norm, "total_norm": total_norm,
            "use_decoupled_grad": decoupled}


def compute_ref(parameters, max_norm, total_norm, use_decoupled_grad):
    return kernelgenbench.baseline.v1_megatron_clip_grad_(
        parameters, max_norm, total_norm, use_decoupled_grad)


def compute_res(parameters, max_norm, total_norm, use_decoupled_grad):
    return kernelgenbench.solution.v1_megatron_clip_grad_(
        parameters, max_norm, total_norm, use_decoupled_grad)


def _cuda_side(config):
    """Independent CUDA parameter list copied from the CPU fixture."""
    params = input_build(config)["parameters"]
    use_decoupled = config[2]
    out = []
    for param in params:
        moved = torch.nn.Parameter(torch.zeros(param.shape, dtype=param.dtype, device="cuda"))
        grad = param.decoupled_grad if use_decoupled else param.grad
        if grad is not None:
            grad = fresh_tensor(grad)
            if use_decoupled:
                moved.decoupled_grad = grad
            else:
                moved.grad = grad
        out.append(moved)
    return out


def _grads(params, use_decoupled):
    return [p.decoupled_grad if use_decoupled else p.grad for p in params]


def compare(ref_out, res_out):
    """Both sides must return None; the write-back is checked on the buffers themselves."""
    assert ref_out is None and res_out is None


def compute_ref_performance(parameters, max_norm, total_norm, use_decoupled_grad, saved):
    """Time the reference entry, restoring each gradient outside the timed region."""
    grads = _grads(parameters, use_decoupled_grad)

    def setup():
        for grad, value in zip(grads, saved):
            if grad is not None:
                grad.copy_(value)

    return measure_cuda(lambda: compute_ref(parameters, max_norm, total_norm, use_decoupled_grad),
                        setup)


def compute_res_performance(parameters, max_norm, total_norm, use_decoupled_grad, saved):
    """Time the solution entry under the same restore protocol."""
    grads = _grads(parameters, use_decoupled_grad)

    def setup():
        for grad, value in zip(grads, saved):
            if grad is not None:
                grad.copy_(value)

    return measure_cuda(lambda: compute_res(parameters, max_norm, total_norm, use_decoupled_grad),
                        setup)


@label("v1_megatron_clip_grad_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_clip_grad_(config):
    max_norm, total_norm, decoupled, dtype, layout, missing = config
    ref_params = _cuda_side(config)
    res_params = _cuda_side(config)
    ref_inputs = dict(parameters=ref_params, max_norm=max_norm, total_norm=total_norm,
                      use_decoupled_grad=decoupled)
    res_inputs = dict(parameters=res_params, max_norm=max_norm, total_norm=total_norm,
                      use_decoupled_grad=decoupled)

    ref_grads = _grads(ref_params, decoupled)
    res_grads = _grads(res_params, decoupled)
    ref_saved = [None if g is None else clone_like(g) for g in ref_grads]
    res_saved = [None if g is None else clone_like(g) for g in res_grads]
    ref_before = [g is not None and g.data_ptr() for g in ref_grads]
    res_before = [g is not None and g.data_ptr() for g in res_grads]

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    coefficient = max_norm / (total_norm + 1e-6)
    for side, grads, saved, ptrs, params in (
            ("reference", ref_grads, ref_saved, ref_before, ref_params),
            ("solution", res_grads, res_saved, res_before, res_params)):
        for i, (grad, original, ptr) in enumerate(zip(grads, saved, ptrs)):
            if original is None:
                assert grad is None, f"{side} created a gradient for missing parameter {i}"
                continue
            # Clipping is in place on the very same buffer, with the same strides.
            assert grad.data_ptr() == ptr, f"{side} replaced gradient buffer {i}"
            assert grad.stride() == original.stride(), f"{side} changed layout of gradient {i}"
            expected = original * coefficient if coefficient < 1.0 else original
            torch.testing.assert_close(grad, expected, rtol=0, atol=0,
                                       msg=f"{side} gradient {i}")
            if coefficient >= 1.0:
                assert torch.equal(grad, original), f"{side} modified gradient {i} without clipping"
            assert grad.dtype == dtype and grad.shape == original.shape
            if decoupled:
                assert params[i].grad is None, f"{side} touched .grad in decoupled mode"

    ref_time = compute_ref_performance(ref_params, max_norm, total_norm, decoupled, ref_saved)
    res_time = compute_res_performance(res_params, max_norm, total_norm, decoupled, res_saved)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
