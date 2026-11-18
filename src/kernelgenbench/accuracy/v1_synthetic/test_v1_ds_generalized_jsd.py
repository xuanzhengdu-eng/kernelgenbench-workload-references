"""Accuracy and performance for v1_ds_generalized_jsd.

Contracts exercised per case: beta endpoints (0/1) and the interpolated branch,
temperature scaling, teacher top-k projection, pointwise clipping, an all-False
response mask, and each of the three float dtypes.  Both sides receive the same
freshly built tensors; the response mask must not be modified.

Reference: ``kernelgenbench.baseline.v1_ds_generalized_jsd`` (tuple-unwrapped form
of DeepSpeed's ``JSDLossOutput``).
Solution: the official fixed-version ``generalized_jsd_loss`` bound through
``kernelgenbench.solution``.
"""
import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


CASES = [
    # beta, temperature, teacher_top_k, pointwise_clip, all-False mask
    (0.0, 1.0, None, None, False),
    (0.5, 1.0, None, None, False),
    (1.0, 0.7, None, None, False),
    (0.3, 0.8, 3, None, False),
    (0.5, 1.0, 5, 0.02, False),
    (0.5, 1.0, None, None, True),
]
COMBOS = [(beta, temperature, top_k, clip, empty, dtype)
          for (beta, temperature, top_k, clip, empty) in CASES
          for dtype in (torch.float32, torch.float16, torch.bfloat16)]


def input_build(config):
    beta, temperature, top_k, clip, empty, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    # [B, T, V] = [2, 3, 11]; the mask has four valid positions unless emptied.
    student = torch.randn((2, 3, 11), dtype=dtype, generator=generator)
    teacher = torch.randn((2, 3, 11), dtype=dtype, generator=generator)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    if empty:
        mask = torch.zeros_like(mask)
    return {
        "student_logits": student,
        "teacher_logits": teacher,
        "response_mask": mask,
        "beta": beta,
        "temperature": temperature,
        "teacher_top_k": top_k,
        "pointwise_clip": clip,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_generalized_jsd(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_generalized_jsd(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and len(ref_out) == 3
    assert isinstance(res_out, tuple) and len(res_out) == 3
    tolerances = {
        torch.float32: (2e-5, 2e-6),
        torch.float16: (3e-3, 2e-3),
        torch.bfloat16: (2e-2, 1e-2),
    }[dtype]
    for ref_value, res_value in zip(ref_out, res_out):
        assert isinstance(res_value, torch.Tensor)
        assert res_value.shape == ref_value.shape
        assert res_value.dtype == ref_value.dtype
        if ref_value.dtype.is_floating_point:
            torch.testing.assert_close(res_value, ref_value, rtol=tolerances[0],
                                       atol=tolerances[1])
        else:
            torch.testing.assert_close(res_value, ref_value, rtol=0, atol=0)


def _measure(fn, inputs, warmup=10, repeats=50):
    for _ in range(warmup):
        fn(**inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn(**inputs)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def compute_ref_performance(inputs):
    return _measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return _measure(compute_res, inputs)


@label("v1_ds_generalized_jsd")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_generalized_jsd(config):
    dtype = config[-1]
    inputs = input_build(config)
    def to_device(values):
        return {name: (value.to(kernelgenbench.device) if isinstance(value, torch.Tensor)
                       else value) for name, value in values.items()}

    ref_inputs = to_device(inputs)
    res_inputs = to_device(inputs)
    saved_mask = res_inputs["response_mask"].clone()

    assert ref_inputs["student_logits"].shape == (2, 3, 11)
    assert res_inputs["student_logits"].dtype == dtype
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # Neither side may modify the response mask or the logits.
    torch.testing.assert_close(res_inputs["response_mask"], saved_mask, rtol=0, atol=0)
    for name in ("student_logits", "teacher_logits"):
        torch.testing.assert_close(res_inputs[name].cpu(), inputs[name], rtol=0, atol=0)

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
