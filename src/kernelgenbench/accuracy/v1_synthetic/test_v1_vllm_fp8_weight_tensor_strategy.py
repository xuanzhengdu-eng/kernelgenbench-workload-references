"""Accuracy and performance for v1_vllm_fp8_weight_tensor_strategy.

Operator: vLLM ``process_fp8_weight_tensor_strategy`` (CUDA path).
Source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L1454-L1480

Contract: the unfused weight is requantized in place shard by shard to the
maximum per-shard scale, then returned as a ``(K, N)`` transpose view that
aliases the input.  A scalar scale or a last scale <= -448 marks an already
fused checkpoint and leaves the data untouched.  ``weight_scale`` is not
modified and ``input_scale`` is passed through by identity.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.quant_rows392_401_checks import (
    fresh, measure, timings,
)


# (n, k, logical_widths, weight scales, has input_scale)
# A scalar scale and the -448 last scale are the two "already fused" markers.
COMBOS = [
    (7, 8, [3, 4], [0.125, 0.25], True),
    (9, 7, [3, 0, 4], [0.125, 0.375, 0.25], False),
    (5, 11, [2, 3], [0.25, -448.0], True),
    (4, 5, [4], 0.5, False),
    (8, 16, [2, 2, 4], [0.5, 0.25, 0.125], True),
    (4, 3, [0, 4], [0.125, 0.25], False),
]


def input_build(config):
    n, k, widths, scales, has_input = config
    values = (((torch.arange(n * k, dtype=torch.float32) * 7 + 3) % 31) - 15).reshape(n, k) * 0.25
    return {
        "weight": values.to(torch.float8_e4m3fn),
        "weight_scale": torch.tensor(scales, dtype=torch.float32),
        "logical_widths": list(widths),
        "input_scale": torch.tensor(0.75, dtype=torch.float32) if has_input else None,
    }


def compute_ref(weight, weight_scale, logical_widths, input_scale=None):
    return kernelgenbench.baseline.v1_vllm_fp8_weight_tensor_strategy(
        weight, weight_scale, logical_widths, input_scale)


def compute_res(weight, weight_scale, logical_widths, input_scale=None):
    return kernelgenbench.solution.v1_vllm_fp8_weight_tensor_strategy(
        weight, weight_scale, logical_widths, input_scale)


def is_fused(weight_scale):
    return weight_scale.ndim == 0 or bool(weight_scale[-1] <= -448)


def expected_weight_bits(inputs):
    """Independent requantization: per-shard dequant, rescale by 1/max, cast."""
    weight = inputs["weight"].view(torch.uint8).clone().view(torch.float8_e4m3fn)
    scale = inputs["weight_scale"]
    max_scale = scale.max()
    if not is_fused(scale):
        start = 0
        for index, width in enumerate(inputs["logical_widths"]):
            if width == 0:
                continue
            end = start + width
            dequant = weight[start:end, :].to(torch.float16) * scale[index]
            weight[start:end, :] = (dequant.float() / max_scale).clamp(
                -448.0, 448.0).to(torch.float8_e4m3fn)
            start = end
    return weight.view(torch.uint8).contiguous()


def compare(ref_out, res_out, inputs):
    assert isinstance(ref_out, tuple) and len(ref_out) == 3
    assert isinstance(res_out, tuple) and len(res_out) == 3
    n, k = inputs["weight"].shape
    expected_weight = expected_weight_bits(inputs)
    expected = expected_weight.view(torch.float8_e4m3fn).t().contiguous().view(torch.uint8)
    for name, out in (("reference", ref_out), ("solution", res_out)):
        assert out[0].dtype == torch.float8_e4m3fn
        assert out[0].shape == (k, n), f"{name} weight must be transposed to (K, N)"
        assert out[0].stride() == (1, k)
        assert torch.equal(out[0].contiguous().view(torch.uint8).cpu(), expected)
        assert out[1].shape == torch.Size([]) and out[1].dtype == torch.float32
        assert torch.equal(out[1].cpu(), inputs["weight_scale"].max())
        assert not torch._C._is_alias_of(out[1], inputs["weight_scale"])
        if inputs["input_scale"] is None:
            assert out[2] is None
        else:
            assert torch.equal(out[2].cpu(), inputs["input_scale"])


@label("v1_vllm_fp8_weight_tensor_strategy")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_fp8_weight_tensor_strategy(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, inputs)

    # In-place requantization must match on both sides and must not touch the
    # scales; the returned weight must alias the mutated input storage.
    expected = expected_weight_bits(inputs)
    for name, out, side in (("reference", ref_out, ref_inputs),
                            ("solution", res_out, res_inputs)):
        assert torch.equal(side["weight"].view(torch.uint8).contiguous().cpu(),
                           expected), f"{name} weight bytes differ after requantization"
        assert out[0].untyped_storage().data_ptr() == \
            side["weight"].untyped_storage().data_ptr()
        assert torch.equal(side["weight_scale"].cpu(), inputs["weight_scale"])
        if side["input_scale"] is not None:
            assert out[2] is side["input_scale"], f"{name} must pass input_scale through"

    return timings(measure(compute_ref, inputs, kernelgenbench.device),
                   measure(compute_res, inputs, kernelgenbench.device))
