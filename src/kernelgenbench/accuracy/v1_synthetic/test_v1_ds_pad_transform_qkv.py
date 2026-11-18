"""Accuracy and performance for v1_ds_pad_transform_qkv.

Official operator: DeepSpeed ``pad_transform_fp16``/``pad_transform_bf16``
(inference transform.cu). ``[B, S, H*D]`` Q/K/V become ``[B, H, S, D]``; with
``add_padding`` D is rounded to 32/64/128 and K/V to a 128-token sequence. The
compiled kernels exist for fp16 and bf16 only.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import upstream_op_test_utils as util
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# batch, heads, query length, KV length, head width, dtype, add_padding
COMBOS = [
    (1, 2, 16, 11, 24, torch.float16, True),
    (1, 2, 32, 16, 32, torch.bfloat16, True),
    (1, 1, 16, 7, 64, torch.float16, True),
    (2, 2, 16, 12, 16, torch.bfloat16, False),
    (2, 4, 8, 8, 128, torch.float16, False),
    (2, 2, 8, 6, 8, torch.bfloat16, True),
    (1, 4, 128, 128, 128, torch.float16, True),
]


def input_build(config):
    batch, heads, q_len, kv_len, head_dim, dtype, padding = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    width = heads * head_dim
    query = torch.randn((batch, q_len, width), generator=generator, dtype=dtype)
    key = torch.randn((batch, kv_len, width), generator=generator, dtype=dtype)
    value = torch.randn((batch, kv_len, width), generator=generator, dtype=dtype)
    return {"query": query, "key": key, "value": value, "heads": heads,
            "add_padding": padding}


def compute_ref(query, key, value, heads, add_padding):
    return kernelgenbench.baseline.v1_ds_pad_transform_qkv(
        query, key, value, heads, add_padding)


def compute_res(query, key, value, heads, add_padding):
    return kernelgenbench.solution.v1_ds_pad_transform_qkv(
        query, key, value, heads, add_padding)


def compare(ref_out, res_out, config):
    batch, heads, q_len, kv_len, head_dim, dtype, padding = config
    assert isinstance(res_out, (list, tuple)) and len(res_out) == 3
    if padding:
        width = 32 if head_dim < 32 else 64 if head_dim < 64 else 128
    else:
        width = head_dim
    expected = [(batch, heads, q_len, width),
                (batch, heads, 128 if padding else kv_len, width),
                (batch, heads, 128 if padding else kv_len, width)]
    for got, want, shape in zip(res_out, ref_out, expected):
        assert got.shape == want.shape == shape, (got.shape, want.shape, shape)
        assert got.dtype == want.dtype == dtype
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return util.measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return util.measure(compute_res, inputs)


@label("v1_ds_pad_transform_qkv")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_pad_transform_qkv(config):
    batch, heads, q_len, kv_len, head_dim, dtype, padding = config
    inputs = input_build(config)
    ref_inputs = util.fresh(inputs, kernelgenbench.device)
    res_inputs = util.fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    # The transform is a pure layout move: real lanes are copied, pads are zero.
    for got, source in ((res_out[0], res_inputs["query"]),
                        (res_out[1], res_inputs["key"]),
                        (res_out[2], res_inputs["value"])):
        lanes = source.shape[1]
        assert got.shape[2] >= lanes
        moved = source.reshape(batch, lanes, heads, head_dim).transpose(1, 2)
        torch.testing.assert_close(got[:, :, :lanes, :head_dim], moved, rtol=0, atol=0)
        if got.shape[2] > lanes or got.shape[3] > head_dim:
            assert torch.count_nonzero(got[:, :, lanes:, :]) == 0
            assert torch.count_nonzero(got[:, :, :lanes, head_dim:]) == 0
        assert not torch._C._is_alias_of(got, source)
    for name in ("query", "key", "value"):
        torch.testing.assert_close(res_inputs[name].cpu(), inputs[name], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)


def test_fp32_is_rejected():
    """The CUDA kernel is only instantiated for fp16/bf16; fp32 entries are stubs."""
    tensor = torch.zeros((1, 8, 16), dtype=torch.float32, device=kernelgenbench.device)
    with pytest.raises(AssertionError):
        kernelgenbench.baseline.v1_ds_pad_transform_qkv(tensor, tensor, tensor, 2, False)
