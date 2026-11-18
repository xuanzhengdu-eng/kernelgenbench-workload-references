"""Accuracy and performance for v1_vllm_weight_amax (723-row 405).

Official operator under test:
``vllm.model_executor.layers.quantization.utils.quant_utils.
weight_amax(weight, *, dim=None, keepdim=False)`` at the pinned vLLM commit
f358f0578a83f4e1fe6fc49a492d3ae9362de8e5.

What the test shows
-------------------
``input_build`` produces real floating weights with values in ``[-4, 4)`` plus
selected special entries (NaN, ``-inf``, all zeros). ``COMBOS`` varies the
reduction axis (``None``, positive, negative), ``keepdim``, the dtype and a
strided (stride 2) layout. Values must match bit-exactly (``equal_nan``), the
shape/dtype must follow ``keepdim`` and the input dtype, and a candidate that
forgets ``abs()`` fails every case that has a negative element.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows402_411_checks import (
    dtype_of, fresh, measure, timings,
)
from sandbox.verifier.test_parametrize import label, parametrize


# (shape, dim, keepdim, strided, special). special is 'none', 'nan', 'inf' or
# 'zero' and seeds one element before the reduction.
CASES = [
    ((3, 7), None, False, False, "none"),
    ((3, 7), 1, True, False, "none"),
    ((3, 7), -2, False, True, "none"),
    ((2, 3, 4), -2, False, True, "none"),
    ((2, 3, 4), 0, True, False, "none"),
    ((4, 3), 0, False, False, "nan"),
    ((2, 5), -1, True, True, "inf"),
    ((1, 1), None, True, False, "zero"),
    ((1, 1), 0, False, False, "zero"),
]
COMBOS = [(case, dtype) for case in CASES
          for dtype in (torch.float32, torch.float16, torch.bfloat16)]


def input_build(config):
    """weight: ``randn*4`` of the requested shape/dtype, plus the seeded edge
    element. A 'strided' entry keeps stride 2 on the last axis."""
    (shape, dim, keepdim, strided, special), dtype = config
    generator = torch.Generator(device="cpu").manual_seed(405)
    weight = torch.randn(shape, generator=generator, dtype=torch.float32) * 4
    flat = weight.reshape(-1)
    if special == "nan":
        flat[0] = float("nan")
    elif special == "inf":
        flat[0] = float("-inf")
    elif special == "zero":
        weight.zero_()
    weight = weight.to(dtype)
    if strided:
        buffer = torch.zeros(shape[:-1] + (shape[-1] * 2,), dtype=dtype)
        buffer[..., ::2] = weight
        weight = buffer[..., ::2]
    return {"weight": weight, "dim": dim, "keepdim": keepdim}


def compute_ref(weight, dim, keepdim):
    return kernelgenbench.baseline.v1_vllm_weight_amax(
        weight, dim=dim, keepdim=keepdim)


def compute_res(weight, dim, keepdim):
    return kernelgenbench.solution.v1_vllm_weight_amax(
        weight, dim=dim, keepdim=keepdim)


def compare(ref_out, res_out, config):
    dtype = dtype_of(config)
    assert isinstance(res_out, torch.Tensor), type(res_out).__name__
    assert res_out.shape == ref_out.shape, \
        f"shape {tuple(res_out.shape)} != {tuple(ref_out.shape)}"
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device.type == ref_out.device.type
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0, equal_nan=True)


def compute_ref_performance(weight, dim, keepdim):
    return measure(compute_ref, {"weight": weight, "dim": dim, "keepdim": keepdim})


def compute_res_performance(weight, dim, keepdim):
    return measure(compute_res, {"weight": weight, "dim": dim, "keepdim": keepdim})


@label("v1_vllm_weight_amax")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_weight_amax(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before = res_inputs["weight"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    torch.testing.assert_close(res_inputs["weight"].cpu(), before.cpu(),
                               rtol=0, atol=0, equal_nan=True)
    assert not torch._C._is_alias_of(res_out, res_inputs["weight"])

    return timings(compute_ref_performance(**ref_inputs),
                   compute_res_performance(**res_inputs))
