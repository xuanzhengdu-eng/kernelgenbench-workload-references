"""vLLM max-scale FP8 requantization: shards, sentinel checkpoint, scalar scale."""
import kernelgenbench
import pytest
import torch
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows422_431_utils import (
    assert_no_alias, fresh, measure)

# (logical_widths, weight_scale values); weight rows = sum(logical_widths), 16 columns.
# Byte-level equality is the contract: the FP8 weight is rewritten in place.
COMBOS = [
    ([2, 3], [0.125, 0.5]),
    ([3, 2], [0.3, 0.7]),
    ([2, 0, 3], [0.125, 0.75, 0.5]),
    ([2, 3], [0.5, -448.0]),
    ([5], 0.5),
    ([8, 8], [0.25, 0.5]),
]


def input_build(config):
    """Deterministic FP8 weight rows with 16 contiguous columns per row."""
    widths, scales = config
    rows = sum(widths)
    raw = ((torch.arange(rows * 16, dtype=torch.float32).reshape(rows, 16) % 49) - 24) * 0.375
    weight = raw.to(torch.float8_e4m3fn)
    scale = torch.tensor(scales, dtype=torch.float32)
    return {"weight": weight, "weight_scale": scale, "logical_widths": list(widths)}


def compute_ref(weight, weight_scale, logical_widths):
    return kernelgenbench.baseline.v1_vllm_fp8_requant_max_(weight, weight_scale, logical_widths)


def compute_res(weight, weight_scale, logical_widths):
    return kernelgenbench.solution.v1_vllm_fp8_requant_max_(weight, weight_scale, logical_widths)


def _expected(weight, weight_scale, logical_widths):
    """Independent per-element dequantize/requantize on the host."""
    out = weight.clone()
    top = weight_scale.max()
    fused = weight_scale.ndim == 0 or not bool(weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min)
    recomputed = 0
    if not fused:
        inv = 1.0 / float(top)
        start = 0
        for index, width in enumerate(logical_widths):
            if width == 0:
                continue
            for row in range(start, start + width):
                for col in range(weight.shape[1]):
                    value = float(weight[row, col].to(torch.float16)) * float(weight_scale[index])
                    level = min(max(value * inv, -448.0), 448.0)
                    out[row, col] = torch.tensor(level, dtype=torch.float32).to(torch.float8_e4m3fn)
            recomputed += width
            start += width
    return out, top, fused, recomputed


def compare(ref_out, res_out, ref_inputs, res_inputs):
    inputs = ref_inputs
    weight, weight_scale = inputs["weight"], inputs["weight_scale"]
    for out in (ref_out, res_out):
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0].shape == weight_scale.max().shape and out[0].dtype == torch.float32
        assert out[1].shape == weight.shape and out[1].dtype == torch.float8_e4m3fn
    # Both sides return the very object they were handed and rewrite its bytes.
    assert ref_out[1] is ref_inputs["weight"], "reference must return the weight object"
    assert res_out[1] is res_inputs["weight"], "solution must return the weight object"
    assert torch.equal(ref_out[0], res_out[0])
    assert torch.equal(ref_out[1].view(torch.uint8), res_out[1].view(torch.uint8))


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_vllm_fp8_requant_max_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_fp8_requant_max_(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    scale_before = inputs["weight_scale"].clone()

    expected, top, fused, recomputed = _expected(inputs["weight"], inputs["weight_scale"], config[0])
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs)

    # Rewritten bytes follow the float32 dequantize/requantize formula.
    assert torch.equal(res_out[1].view(torch.uint8).cpu(), expected.view(torch.uint8)), "requantized bytes differ"
    assert torch.equal(res_out[0].cpu(), top), "returned max scale differs"
    assert (recomputed == 0) == fused, "sentinel/scalar path must not requantize"

    # Only the weight is rewritten; the scale tensor is read-only for the op.
    torch.testing.assert_close(res_inputs["weight_scale"].cpu(), scale_before, rtol=0, atol=0)
    assert_no_alias(res_out[0], res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
