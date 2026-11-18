"""Accuracy and performance for v1_clip_tensors_global_norm_.

DeepSpeed ``clip_tensors_by_global_norm`` with a caller-supplied global norm:
every tensor is multiplied in place by ``max_norm / (global_norm + eps)`` when
that coefficient is below one, and the supplied norm is returned unchanged (the
same tensor object when a tensor was passed).  Empty tensors are legal.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows172_181_test_utils import fresh_tensor, measure_cuda
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (global_norm, max_norm, norm_is_tensor, shapes, eps)
COMBOS = [
    (8.0, 1.0, True, ((4, 8), (3,)), 1e-6),
    (0.5, 1.0, True, ((5,), (2, 3)), 1e-6),
    (1.0, 1.0, False, ((4, 4),), 1e-6),
    (0.0, 1.0, True, ((0,),), 1e-6),
    (2.0, 1.0, True, ((3, 5), (2, 2)), 0.25),
    (8.0, 1.0, False, ((6,), (3, 2)), 1e-6),
]


def input_build(config, dtype):
    norm, max_norm, norm_is_tensor, shapes, eps = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    tensors = [torch.randn(shape, dtype=dtype, generator=generator) for shape in shapes]
    value = torch.tensor(norm, dtype=torch.float32) if norm_is_tensor else norm
    return {"input_tensors": tensors, "max_norm": max_norm, "global_norm": value, "eps": eps}


def compute_ref(input_tensors, max_norm, global_norm, eps):
    return kernelgenbench.baseline.v1_clip_tensors_global_norm_(
        input_tensors, max_norm, global_norm, eps)


def compute_res(input_tensors, max_norm, global_norm, eps):
    return kernelgenbench.solution.v1_clip_tensors_global_norm_(
        input_tensors, max_norm, global_norm, eps)


def compare(ref_out, res_out, res_norm):
    if isinstance(ref_out, torch.Tensor):
        assert isinstance(res_out, torch.Tensor)
        assert res_out is res_norm, "solution must return the supplied norm object"
        assert torch.equal(ref_out.cpu(), res_out.cpu())
    else:
        assert type(res_out) is float and ref_out == res_out
        torch.testing.assert_close(torch.tensor(ref_out), torch.tensor(res_out), rtol=0, atol=0)


def compute_ref_performance(tensors, max_norm, global_norm, eps, saved):
    """Time the reference entry, restoring every tensor outside the timed region."""
    def setup():
        for tensor, value in zip(tensors, saved):
            tensor.copy_(value)

    return measure_cuda(lambda: compute_ref(tensors, max_norm, global_norm, eps), setup)


def compute_res_performance(tensors, max_norm, global_norm, eps, saved):
    """Time the solution entry under the same restore protocol."""
    def setup():
        for tensor, value in zip(tensors, saved):
            tensor.copy_(value)

    return measure_cuda(lambda: compute_res(tensors, max_norm, global_norm, eps), setup)


@label("v1_clip_tensors_global_norm_")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("config", COMBOS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_accuracy_v1_clip_tensors_global_norm_(config, dtype):
    norm, max_norm, norm_is_tensor, shapes, eps = config
    cpu_inputs = input_build(config, dtype)

    ref_tensors = [fresh_tensor(t) for t in cpu_inputs["input_tensors"]]
    res_tensors = [fresh_tensor(t) for t in cpu_inputs["input_tensors"]]
    ref_norm = (fresh_tensor(cpu_inputs["global_norm"])
                if norm_is_tensor else cpu_inputs["global_norm"])
    res_norm = (fresh_tensor(cpu_inputs["global_norm"])
                if norm_is_tensor else cpu_inputs["global_norm"])
    ref_ptrs = [t.data_ptr() for t in ref_tensors]
    res_ptrs = [t.data_ptr() for t in res_tensors]

    ref_out = compute_ref(ref_tensors, max_norm, ref_norm, eps)
    res_out = compute_res(res_tensors, max_norm, res_norm, eps)
    compare(ref_out, res_out, res_norm)

    coefficient = max_norm / (float(norm) + eps)
    # Both sides run the same official arithmetic, so they must agree bit for bit.
    for x, z in zip(ref_tensors, res_tensors):
        assert torch.equal(x, z), "reference and solution disagree"
    tolerance = {torch.float32: 1e-6, torch.bfloat16: 8e-3}[dtype]
    for side, tensors, ptrs in (("reference", ref_tensors, ref_ptrs),
                                ("solution", res_tensors, res_ptrs)):
        for i, (tensor, ptr) in enumerate(zip(tensors, ptrs)):
            original = cpu_inputs["input_tensors"][i]
            assert tensor.data_ptr() == ptr, f"{side} replaced tensor {i}"
            assert tensor.dtype == dtype and tensor.shape == original.shape
            expected = original * coefficient if coefficient < 1.0 else original
            torch.testing.assert_close(tensor.cpu(), expected, rtol=tolerance, atol=tolerance,
                                       msg=f"{side} tensor {i}")
            if coefficient >= 1.0:
                assert torch.equal(tensor.cpu(), original), f"{side} changed tensor {i} without clipping"

    ref_saved = [t.clone() for t in ref_tensors]
    res_saved = [t.clone() for t in res_tensors]

    ref_time = compute_ref_performance(ref_tensors, max_norm, ref_norm, eps, ref_saved)
    res_time = compute_res_performance(res_tensors, max_norm, res_norm, eps, res_saved)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
