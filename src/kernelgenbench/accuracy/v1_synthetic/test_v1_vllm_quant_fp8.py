"""Accuracy and performance for v1_vllm_quant_fp8 (QuantFP8.forward_native).

Item: v1_vllm_quant_fp8 (723-row 414).  The cache ``convert_fp8`` branch of the
old reference has a different signature/return/mutation and is split into
``v1_vllm_cache_convert_fp8``.
Official source (pinned commit f358f057):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/input_quant_fp8.py#L184-L267

Configs: ``(x_shape, group_shape, scale_shape, scale_ub, token_padding,
use_ue8m0, column_major_scales)``.  ``group_shape=(1, g)`` is dynamic per-token
group quantization; ``(1, -1)`` per token; ``(-1, -1)`` per tensor; a static
1D scale with ``(-1, 4)`` broadcasts over four-column groups.  Quantized bytes
and the returned scale tensor must match exactly.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows412_421_checks as checks
from sandbox.verifier.test_parametrize import label, parametrize


CASES = [
    ((4, 16), (-1, -1), None, None, None, False, False),
    ((4, 16), (1, -1), None, 3.0, 8, False, False),
    ((4, 16), (-1, -1), (1,), None, None, False, False),
    ((4, 16), (-1, 4), (4,), None, 8, False, False),
    ((4, 16), (1, -1), (4,), None, None, False, False),
    ((4, 18), (1, 8), None, None, None, False, False),
    ((7, 31), (1, 8), None, None, None, True, True),
]
COMBOS = [(case, dtype) for case in CASES for dtype in (torch.float32, torch.float16, torch.bfloat16)]


def input_build(config):
    case, dtype = config
    shape, group_shape, scale_shape, scale_ub, padding, use_ue8m0, column_major = case
    device = getattr(kernelgenbench, "device", torch.device("cpu"))
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    x = torch.randn(shape, dtype=dtype, generator=generator)
    scale = None if scale_shape is None else torch.rand(scale_shape, dtype=torch.float32, generator=generator) + 0.1
    upper = None if scale_ub is None else torch.tensor(scale_ub, dtype=torch.float32)
    if torch.device(device).type == "cuda":
        x = x.to(device)
        scale = scale.to(device) if scale is not None else None
        upper = upper.to(device) if upper is not None else None
    return {
        "x": x,
        "group_shape": group_shape,
        "scale": scale,
        "scale_ub": upper,
        "num_token_padding": padding,
        "use_ue8m0": use_ue8m0,
        "column_major_scales": column_major,
    }


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_vllm_quant_fp8(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_vllm_quant_fp8(**kwargs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    ref_q, ref_s = ref_out
    res_q, res_s = res_out
    assert res_q.shape == ref_q.shape and res_q.dtype == ref_q.dtype == torch.float8_e4m3fn
    assert ref_s.shape == res_s.shape and ref_s.dtype == res_s.dtype == torch.float32
    assert torch.equal(ref_q.view(torch.uint8).cpu(), res_q.view(torch.uint8).cpu())
    assert torch.equal(ref_s.cpu(), res_s.cpu())


@label("v1_vllm_quant_fp8")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_quant_fp8(config):
    case, dtype = config
    device = kernelgenbench.device
    ref_inputs = input_build(config)
    res_inputs = input_build(config)
    ref_snapshot = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in ref_inputs.items()}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    for name, value in res_inputs.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, ref_snapshot[name]), f"input {name} was modified"
    # The quantized payload is always fresh.  A static scale is returned as a
    # view of the caller's scale tensor, matching forward_native.
    for value in res_inputs.values():
        if isinstance(value, torch.Tensor):
            assert not torch._C._is_alias_of(res_out[0], value), "quantized output aliases an input"
    if res_inputs["scale"] is None:
        for value in res_inputs.values():
            if isinstance(value, torch.Tensor):
                assert not torch._C._is_alias_of(res_out[1], value), "dynamic scale aliases an input"

    ref_time = checks.measure(compute_ref, lambda: input_build(config), device)
    res_time = checks.measure(compute_res, lambda: input_build(config), device)
    return checks.timings(ref_time, res_time)
