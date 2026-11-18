"""Accuracy and timing for v1_sgl_qkvzba_contiguous_cpu.

Contiguous input is [all q | all k | all v | all z] and [all b | all a].
Unlike the grouped entry, the native code does not require a head-group
ratio or equal QK/V head widths; the test covers differing widths.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows642_651_test_utils import (
    aliases,
    clone_tree,
    measure_ms,
    move_tree,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (batch, num_heads_qk, num_heads_v, head_qk, head_v, dtype)
COMBOS = [
    (3, 2, 4, 4, 6, torch.bfloat16),
    (1, 1, 2, 8, 8, torch.float16),
    (2, 3, 2, 5, 7, torch.bfloat16),
    (4, 2, 3, 6, 2, torch.float16),
    (2, 2, 4, 6, 6, torch.bfloat16),
]


def input_build(config):
    batch, num_heads_qk, num_heads_v, head_qk, head_v, dtype = config
    generator = torch.Generator().manual_seed(645)
    mixed_qkvz = torch.randn(
        (batch, 2 * num_heads_qk * head_qk + 2 * num_heads_v * head_v),
        dtype=dtype,
        generator=generator,
    )
    mixed_ba = torch.randn(
        (batch, 2 * num_heads_v), dtype=dtype, generator=generator
    )
    return {
        "mixed_qkvz": mixed_qkvz,
        "mixed_ba": mixed_ba,
        "num_heads_qk": num_heads_qk,
        "num_heads_v": num_heads_v,
        "head_qk": head_qk,
        "head_v": head_v,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_qkvzba_contiguous_cpu(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_qkvzba_contiguous_cpu(**inputs)


def compare(ref_out, res_out, config):
    batch, num_heads_qk, num_heads_v, head_qk, head_v, dtype = config
    qkv_dim = 2 * num_heads_qk * head_qk + num_heads_v * head_v
    assert isinstance(ref_out, tuple) and len(ref_out) == 4
    assert isinstance(res_out, tuple) and len(res_out) == 4
    expected_shapes = (
        (batch, qkv_dim),
        (batch, num_heads_v, head_v),
        (batch, num_heads_v),
        (batch, num_heads_v),
    )
    for actual, expected, shape in zip(res_out, ref_out, expected_shapes):
        assert actual.shape == expected.shape == shape
        assert actual.dtype == expected.dtype == dtype
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return measure_ms(compute_ref, inputs, torch.device("cpu"), repetitions=30)


def compute_res_performance(**inputs):
    return measure_ms(compute_res, inputs, torch.device("cpu"), repetitions=30)


@label("v1_sgl_qkvzba_contiguous_cpu")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_qkvzba_contiguous_cpu(config):
    inputs = input_build(config)
    ref_inputs = move_tree(clone_tree(inputs), torch.device("cpu"))
    res_inputs = move_tree(clone_tree(inputs), torch.device("cpu"))
    ref_before = clone_tree(ref_inputs)
    res_before = clone_tree(res_inputs)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    for output in ref_out:
        for name in ("mixed_qkvz", "mixed_ba"):
            assert not aliases(output, ref_inputs[name])
    for output in res_out:
        for name in ("mixed_qkvz", "mixed_ba"):
            assert not aliases(output, res_inputs[name])
    for name in ("mixed_qkvz", "mixed_ba"):
        torch.testing.assert_close(
            ref_inputs[name], ref_before[name], rtol=0, atol=0
        )
        torch.testing.assert_close(
            res_inputs[name], res_before[name], rtol=0, atol=0
        )

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time,
        res_time=res_time,
        speedup=ref_time / res_time if res_time else float("inf"),
    )
