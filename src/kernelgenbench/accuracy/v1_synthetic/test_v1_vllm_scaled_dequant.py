"""Accuracy and performance for v1_vllm_scaled_dequant (723-row 411, native item).

Official operator under test:
``vllm.model_executor.layers.quantization.utils.quant_utils.scaled_dequantize``
at the pinned vLLM commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5.

What the test shows
-------------------
``input_build`` makes a ``float8_e4m3`` matrix (2-D, plus one 1-D vector) and an
fp32 scale in one of the supported scale shapes, together with the real vLLM
``GroupShape`` for the grouped cases. ``COMBOS`` covers a per-tensor scalar, a
per-tensor group shape, a per-row scale (group ``(M,1)``), a per-column scale
(group ``(1,N)``), a 4x4 block scale with group ``(2,3)`` (each scale is
repeated 4x4), a 2-D scale without a group shape, a 1-D input (broadcast to
``(1, N)``), and the ``float32``/``float16``/``bfloat16`` output dtypes. The test asserts shape, dtype, device and bit-exact values against the
baseline, plus that neither ``x_q`` nor ``x_s`` was modified and that the result
is a fresh tensor. The cache-writing ``convert_fp8`` branch is a different
logical entry and has its own item, ``v1_vllm_cache_dequant_fp8``.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows402_411_checks import (
    fresh, measure, timings,
)
from sandbox.verifier.test_parametrize import label, parametrize


# (x_q shape, group shape as (row, col) with -1 = full extent, scale shape).
# A 1-D x_q is legal in the official entry: the broadcast returns (1, N), which
# is asserted explicitly rather than being described as a 2-D-only contract.
CASES = [
    ((4, 8), None, (1,)),
    ((4, 8), (4, 8), (1,)),
    ((4, 8), (4, 2), (4,)),
    ((4, 8), (1, 8), (4,)),
    ((8, 12), (2, 3), (4, 4)),
    ((8, 12), None, (2, 3)),
    ((16, 256), (1, 128), (16, 2)),
    ((16, 256), (16, 128), (1, 2)),
    ((8,), None, (1,)),
]
COMBOS = [(case, dtype) for case in CASES
          for dtype in (torch.float32, torch.float16, torch.bfloat16)]


def input_build(config):
    """x_q: fp8_e4m3 values in [-4, 4); x_s: fp32 positive scales.

    The group shape is the real ``GroupShape`` namedtuple from vLLM (imported
    lazily so this module stays importable in the other framework images).
    """
    x_shape, group, scale_shape = config[0]
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    generator = torch.Generator(device="cpu").manual_seed(411)
    matrix = torch.randn(x_shape, generator=generator).to(torch.float8_e4m3fn)
    scale = torch.rand(scale_shape, generator=generator) + 0.1
    return {"x_q": matrix, "x_s": scale,
            "group_shape": GroupShape(*group) if group is not None else None}


def compute_ref(x_q, x_s, group_shape, out_dtype):
    return kernelgenbench.baseline.v1_vllm_scaled_dequant(
        x_q, x_s, group_shape=group_shape, out_dtype=out_dtype)


def compute_res(x_q, x_s, group_shape, out_dtype):
    return kernelgenbench.solution.v1_vllm_scaled_dequant(
        x_q, x_s, group_shape=group_shape, out_dtype=out_dtype)


def compare(ref_out, res_out, config):
    (x_shape, _group, _shape), dtype = config
    # A 2-D matrix keeps its shape; a 1-D vector comes back with a leading 1.
    expected = tuple(x_shape) if len(x_shape) == 2 else (1, *x_shape)
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == expected
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == ref_out.device
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_vllm_scaled_dequant")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_scaled_dequant(config):
    dtype = config[1]
    inputs = input_build(config)
    inputs["out_dtype"] = dtype
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before_q = res_inputs["x_q"].clone()
    before_s = res_inputs["x_s"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    assert torch.equal(res_inputs["x_q"].view(torch.uint8), before_q.view(torch.uint8))
    torch.testing.assert_close(res_inputs["x_s"], before_s, rtol=0, atol=0)
    assert not torch._C._is_alias_of(res_out, res_inputs["x_q"])

    return timings(compute_ref_performance(**ref_inputs),
                   compute_res_performance(**res_inputs))
