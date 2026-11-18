"""Accuracy and performance for v1_sgl_vit_dummy_head_pad.

Official operator: SGLang ``pad_vit_attn_dummy_heads`` (vision_utils.py). The
solution binding adapts the five scalars to the upstream config object, so the
official function receives ``config.vision_config.{num_dummy_heads, head_dim}``.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import upstream_op_test_utils as util
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# name, shape, head dimension, dummy heads, strided input, dtype
COMBOS = [
    ("attn.qkv_proj.weight", (24, 5), 4, 1, False, torch.float32),
    ("attn.qkv_proj.bias", (18,), 3, 2, False, torch.float32),
    ("attn.qkv_proj.weight", (96, 8), 8, 3, False, torch.bfloat16),
    ("attn.q_proj.weight", (8, 5), 4, 1, True, torch.float32),
    ("attn.k_proj.bias", (6,), 3, 1, False, torch.float16),
    ("attn.v_proj.weight", (6, 4), 3, 0, False, torch.float32),
    ("attn.proj.weight", (5, 8), 4, 2, True, torch.float32),
    ("attn.q_norm.weight", (4,), 4, 1, False, torch.float32),
    ("attn.k_norm.weight", (8,), 4, 1, False, torch.float32),
    ("unrelated.weight", (6, 4), 2, 1, False, torch.float32),
]
INVALID = [
    ("attn.qkv_proj.scale", torch.zeros(24, 5)),
    ("attn.q_proj.scale", torch.zeros(8, 5)),
]


def input_build(config):
    name, shape, head_dim, dummy_heads, strided, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    raw = (*shape[:-1], shape[-1] * 2) if strided else shape
    values = torch.randn(raw, generator=generator, dtype=torch.float32).to(dtype)
    if strided:
        values = values[..., ::2]
    return {"loaded_weight": values, "name": name, "head_dim": head_dim,
            "num_dummy_heads": dummy_heads}


def compute_ref(loaded_weight, name, head_dim, num_dummy_heads):
    return kernelgenbench.baseline.v1_sgl_vit_dummy_head_pad(
        loaded_weight, name, head_dim, num_dummy_heads)


def compute_res(loaded_weight, name, head_dim, num_dummy_heads):
    return kernelgenbench.solution.v1_sgl_vit_dummy_head_pad(
        loaded_weight, name, head_dim, num_dummy_heads)


def compare(ref_out, res_out, config):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return util.measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return util.measure(compute_res, inputs)


@label("v1_sgl_vit_dummy_head_pad")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_vit_dummy_head_pad(config):
    name, shape, head_dim, dummy_heads, strided, dtype = config
    inputs = input_build(config)
    ref_inputs = util.fresh(inputs, kernelgenbench.device)
    res_inputs = util.fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    weight = res_inputs["loaded_weight"]
    extra = head_dim * dummy_heads
    passthrough = (dummy_heads == 0 or "unrelated" in name
                   or (("q_norm" in name or "k_norm" in name) and weight.numel() == head_dim))
    if passthrough:
        assert res_out is weight, "unmodified weights must be returned unchanged"
    elif "qkv_proj" in name:
        part = shape[0] // 3
        block = part + extra
        assert res_out.shape[0] == 3 * block
        for index in range(3):
            torch.testing.assert_close(res_out[index * block:index * block + part],
                                       weight[index * part:(index + 1) * part],
                                       rtol=0, atol=0)
            assert torch.count_nonzero(res_out[index * block + part:(index + 1) * block]) == 0
    elif any(token in name for token in ("attn.q_proj", "attn.k_proj", "attn.v_proj")):
        grown = shape[0] + extra
        assert res_out.shape[0] == grown
        torch.testing.assert_close(res_out[:shape[0]], weight, rtol=0, atol=0)
        assert torch.count_nonzero(res_out[shape[0]:]) == 0
    elif "attn.proj.weight" in name:
        assert res_out.shape == (shape[0], shape[1] + extra)
        torch.testing.assert_close(res_out[:, :shape[1]], weight, rtol=0, atol=0)
        assert torch.count_nonzero(res_out[:, shape[1]:]) == 0
    elif "attn.q_norm.weight" in name or "attn.k_norm.weight" in name:
        grown = shape[0] + extra
        assert res_out.shape == (grown,)
        torch.testing.assert_close(res_out[:shape[0]], weight, rtol=0, atol=0)
        assert torch.count_nonzero(res_out[shape[0]:]) == 0
    else:
        raise AssertionError(f"unclassified weight name {name}")
    if not passthrough:
        assert not torch._C._is_alias_of(res_out, weight), "a padded weight is a new tensor"
    torch.testing.assert_close(res_inputs["loaded_weight"].cpu(), inputs["loaded_weight"],
                               rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)


def test_invalid_weight_names_raise():
    """Non-.weight/.bias names on the padded qkv/q/k/v branches are rejected."""
    device = kernelgenbench.device
    for name, values in INVALID:
        with pytest.raises(RuntimeError):
            kernelgenbench.baseline.v1_sgl_vit_dummy_head_pad(
                values.to(device), name, head_dim=4, num_dummy_heads=1)
