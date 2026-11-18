"""DeepSpeed packed SwiGLU backward; row 28, OP-bb5f17bc3006.

The packed Triton op requires a [..., 2I] tensor and returns one packed [..., 2I]
gradient. It shares this even-width contract with the Megatron elementwise
helper, so both are references of the same Item. Cases are even widths and three
dtypes.
The solution autograds the upstream swiglu_packed Function, so the real Triton
forward and backward kernels run.
"""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.activation_rows22_31_checks import (
    config_dtype, fresh, measure, timings,
)
from sandbox.verifier.test_parametrize import label, parametrize


# (y_shape, dtype); DeepSpeed's packed contract requires an even last width.
COMBOS = [
    (shape, dtype)
    for shape in [(2, 16, 128), (2, 128, 1024), (4, 64)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]


def input_build(config):
    """y [..., 2I] and g [..., I], both even-width packed inputs."""
    shape, dtype = config
    assert shape[-1] % 2 == 0
    out_w = shape[-1] // 2
    rng = torch.Generator(device="cpu").manual_seed(2802)
    y = torch.randn(shape, dtype=dtype, generator=rng)
    g = torch.randn(shape[:-1] + (out_w,), dtype=dtype, generator=rng)
    return {"g": g, "y": y}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_swiglu_back_deepspeed_packed(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_swiglu_back_deepspeed_packed(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = {
        torch.float32: (4e-6, 2e-6),
        torch.float16: (3e-3, 2e-3),
        torch.bfloat16: (2e-2, 1.2e-2),
    }[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_swiglu_back_deepspeed_packed")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_swiglu_back_deepspeed_packed(config):
    shape, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    assert ref_out.shape == shape
    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["g"].cpu(), inputs["g"], rtol=0, atol=0)
        torch.testing.assert_close(side["y"].cpu(), inputs["y"], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
