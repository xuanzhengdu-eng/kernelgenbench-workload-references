"""Accuracy and performance for v1_sgl_mxfp4_moe_deinterleave_.

Row 372. Official source:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/quantization/marlin_utils_fp4.py#L334-L370

The layer holds GPT-OSS gate/up rows interleaved on dim 1. Both sides must turn
``[gate0, up0, gate1, up1, ...]`` into ``[gate0, gate1, ..., up0, up1, ...]``
in place: return ``None``, keep the Python tensor objects, replace their
``.data`` storage and leave every value outside the permutation untouched.

Each COMBOS entry is ``(experts, rows, hidden, scale_field, bias_field)``.
``scale_field``/``bias_field`` select the optional parameter name that the
official loader uses; ``None`` means the layer has no such tensor.
"""
import statistics
import time
import types

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

NAME = "v1_sgl_mxfp4_moe_deinterleave_"

# rows must be even; hidden covers small odd/even widths; all optional-field
# spellings are exercised at least once.
COMBOS = [
    (1, 4, 3, None, None),
    (2, 8, 4, "w13_weight_scale", "w13_weight_bias"),
    (3, 6, 2, "w13_weight_scale_inv", "w13_bias"),
    (1, 8, 8, "w13_weight_scale", None),
    (2, 2, 1, None, "w13_bias"),
    (2, 16, 5, "w13_weight_scale_inv", None),
]

_WEIGHT_KEYS = ("w13_weight",)
_SCALE_KEYS = ("w13_weight_scale", "w13_weight_scale_inv")
_BIAS_KEYS = ("w13_weight_bias", "w13_bias")


def input_build(config):
    """CPU tensors with a value that identifies (expert, row, column)."""
    experts, rows, hidden, scale_field, bias_field = config
    layer = types.SimpleNamespace()
    # row index dominates so a wrong row permutation is visible in every byte.
    index = torch.arange(experts * rows * hidden, dtype=torch.int64)
    layer.w13_weight = (index % 251).to(torch.uint8).reshape(experts, rows, hidden)
    if scale_field is not None:
        setattr(
            layer,
            scale_field,
            (torch.arange(experts * rows * 3, dtype=torch.float32).reshape(experts, rows, 3) + 0.5),
        )
    if bias_field is not None:
        setattr(
            layer,
            bias_field,
            (torch.arange(experts * rows, dtype=torch.float32).reshape(experts, rows) * 0.25 - 3.0),
        )
    return {"layer": layer}


def _fresh(inputs, device):
    """Deep-copy every layer attribute to ``device``; aliases stay independent."""
    layer = types.SimpleNamespace()
    for name, value in vars(inputs["layer"]).items():
        setattr(layer, name, value.to(device=device, copy=True))
    return {"layer": layer}


def compute_ref(layer):
    return kernelgenbench.baseline.v1_sgl_mxfp4_moe_deinterleave_(layer)


def compute_res(layer):
    return kernelgenbench.solution.v1_sgl_mxfp4_moe_deinterleave_(layer)


def _deinterleaved(reference):
    """Independent oracle: even-index rows first, then odd-index rows."""
    return torch.cat((reference[:, ::2], reference[:, 1::2]), dim=1)


def _measure(fn, config):
    """Median ms; the in-place input is rebuilt outside the timed region."""
    device = kernelgenbench.device
    samples = []
    for _ in range(7):
        arguments = _fresh(input_build(config), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(**arguments)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
        assert result is None
    return statistics.median(samples[1:])


@label(NAME)
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_mxfp4_moe_deinterleave_(config):
    experts, rows, hidden, scale_field, bias_field = config
    inputs = input_build(config)
    ref_inputs = _fresh(inputs, kernelgenbench.device)
    res_inputs = _fresh(inputs, kernelgenbench.device)

    keys = ["w13_weight"]
    if scale_field is not None:
        keys.append(scale_field)
    if bias_field is not None:
        keys.append(bias_field)

    # Snapshot pre-call values and object identity for both sides.
    sides = {"ref": ref_inputs, "res": res_inputs}
    before = {
        tag: {name: getattr(side["layer"], name).clone() for name in keys}
        for tag, side in sides.items()
    }
    objects = {
        tag: {name: getattr(side["layer"], name) for name in keys}
        for tag, side in sides.items()
    }

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)

    # Both sides return None and rewrite the same tensor objects' .data.
    assert ref_out is None and res_out is None
    for tag, side in sides.items():
        layer = side["layer"]
        assert layer.w13_weight.shape == (experts, rows, hidden)
        for name in keys:
            tensor = getattr(layer, name)
            assert tensor is objects[tag][name], f"{name}: tensor object was replaced"
            assert tensor.is_contiguous()
            assert tensor.shape == before[tag][name].shape
            assert tensor.dtype == before[tag][name].dtype

    # The weight and every optional tensor follow the gate/up split of dim 1.
    for name in keys:
        torch.testing.assert_close(
            getattr(ref_inputs["layer"], name),
            _deinterleaved(before["ref"][name]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            getattr(res_inputs["layer"], name),
            _deinterleaved(before["res"][name]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            getattr(res_inputs["layer"], name),
            getattr(ref_inputs["layer"], name),
            rtol=0,
            atol=0,
        )

    ref_time = _measure(compute_ref, config)
    res_time = _measure(compute_res, config)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
