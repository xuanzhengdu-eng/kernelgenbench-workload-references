"""Accuracy and performance for v1_zenflow_unscale_clip_.

ZenFlow ``unscale_and_clip_grads`` over one flat fp32 gradient buffer: when the
clip threshold is positive the multiplier is
``1 / (clamp(((total_norm / loss_scale) + 1e-6) / clip_grad, min=1) * loss_scale)``,
otherwise ``1 / loss_scale``.  The buffer is updated in place and the call
returns None.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows172_181_test_utils import fresh_tensor, measure_cuda
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (shape, scaled global norm, loss scale, clipping threshold)
COMBOS = [
    ((4, 8), 64.0, 16.0, 1.0),
    ((8,), 4.0, 16.0, 1.0),
    ((2, 3), 128.0, 32.0, 0.0),
    ((0,), 0.0, 4.0, 1.0),
    ((16, 16), 1.0, 8.0, 0.5),
    ((3, 5), 1024.0, 64.0, 0.25),
]


def input_build(config):
    shape, norm, loss_scale, clip_grad = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    return {"grad": torch.randn(shape, dtype=torch.float32, generator=generator),
            "total_norm": torch.tensor(norm, dtype=torch.float32),
            "loss_scale": loss_scale, "clip_grad": clip_grad}


def compute_ref(grad, total_norm, loss_scale, clip_grad):
    return kernelgenbench.baseline.v1_zenflow_unscale_clip_(
        grad, total_norm, loss_scale, clip_grad)


def compute_res(grad, total_norm, loss_scale, clip_grad):
    return kernelgenbench.solution.v1_zenflow_unscale_clip_(
        grad, total_norm, loss_scale, clip_grad)


def compare(ref_out, res_out):
    assert ref_out is None and res_out is None


def _multiplier(norm, loss_scale, clip_grad):
    if clip_grad > 0:
        combined = torch.clamp(torch.tensor(((norm / loss_scale) + 1e-6) / clip_grad),
                               min=1.0) * loss_scale
    else:
        combined = loss_scale
    return 1.0 / combined


def compute_ref_performance(grad, total_norm, loss_scale, clip_grad, saved):
    """Time the reference entry, restoring the gradient buffer outside the timed region."""
    def setup():
        grad.copy_(saved)

    return measure_cuda(lambda: compute_ref(grad, total_norm, loss_scale, clip_grad), setup)


def compute_res_performance(grad, total_norm, loss_scale, clip_grad, saved):
    """Time the solution entry under the same restore protocol."""
    def setup():
        grad.copy_(saved)

    return measure_cuda(lambda: compute_res(grad, total_norm, loss_scale, clip_grad), setup)


@label("v1_zenflow_unscale_clip_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_zenflow_unscale_clip_(config):
    shape, norm, loss_scale, clip_grad = config
    cpu_inputs = input_build(config)

    ref_grad = fresh_tensor(cpu_inputs["grad"])
    res_grad = fresh_tensor(cpu_inputs["grad"])
    ref_norm = fresh_tensor(cpu_inputs["total_norm"])
    res_norm = fresh_tensor(cpu_inputs["total_norm"])
    ref_ptr, res_ptr = ref_grad.data_ptr(), res_grad.data_ptr()

    ref_out = compute_ref(ref_grad, ref_norm, loss_scale, clip_grad)
    res_out = compute_res(res_grad, res_norm, loss_scale, clip_grad)
    compare(ref_out, res_out)

    expected = cpu_inputs["grad"] * _multiplier(norm, loss_scale, clip_grad)
    for side, grad, ptr, norm_tensor, source in (
            ("reference", ref_grad, ref_ptr, ref_norm, cpu_inputs["total_norm"]),
            ("solution", res_grad, res_ptr, res_norm, cpu_inputs["total_norm"])):
        assert grad.data_ptr() == ptr, f"{side} replaced the gradient buffer"
        assert grad.shape == shape and grad.dtype == torch.float32
        torch.testing.assert_close(grad.cpu(), expected, rtol=1e-6, atol=1e-6,
                                   msg=f"{side} gradient")
        # The norm is read, not modified.
        assert torch.equal(norm_tensor.cpu(), source), f"{side} modified total_norm"

    ref_saved = cpu_inputs["grad"].cuda()
    res_saved = cpu_inputs["grad"].cuda()

    ref_time = compute_ref_performance(ref_grad, ref_norm, loss_scale, clip_grad, ref_saved)
    res_time = compute_res_performance(res_grad, res_norm, loss_scale, clip_grad, res_saved)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
