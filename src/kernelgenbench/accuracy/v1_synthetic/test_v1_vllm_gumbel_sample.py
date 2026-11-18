"""Accuracy and performance for v1_vllm_gumbel_sample.

The six cases cover one-token vocabularies, sub- and super-block vocabularies,
all three logit dtypes, both ``apply_temperature`` settings and draft/target
noise streams. Row mappings include -1 (invalid request) and temperature 0/1.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.sampling_test_utils import (
    assert_unchanged,
    measure_call,
    move,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# rows, vocab, dtype, apply_temperature, is_drafting, logits pattern.
CASES = [
    (4, 17, torch.float32, True, False, "mixed"),
    (6, 1023, torch.float16, False, False, "mixed"),
    (5, 1025, torch.bfloat16, True, True, "mixed"),
    (3, 2049, torch.float32, False, True, "uniform"),
    (4, 1, torch.float32, True, False, "uniform"),
    (8, 64, torch.bfloat16, False, False, "uniform"),
]
COMBOS = [(case,) for case in CASES]


def input_build(config):
    (rows, vocab, dtype, apply_temperature, is_drafting, pattern) = config[0]
    x = torch.arange(rows * vocab, dtype=torch.float32).reshape(rows, vocab)
    if pattern == "mixed":
        logits = ((((x * 13 + 7) % 53) - 26) * 0.03125).to(dtype)
    else:
        logits = torch.zeros_like(x).to(dtype)
    mapping = torch.tensor(
        [(-1 if i % 5 == 4 else (i % 3)) for i in range(rows)], dtype=torch.int32
    )
    return {
        "logits": logits,
        "expanded_idx_mapping": mapping,
        "temperature": torch.tensor([0.5, 0.0, 1.75], dtype=torch.float32),
        # Exercises both 32-bit halves and a negative seed.
        "seed": torch.tensor([17, 2**40 + 19, -123456789], dtype=torch.int64),
        "pos": torch.arange(rows, dtype=torch.int64) * 13 + 7,
        "apply_temperature": apply_temperature,
        "is_drafting": is_drafting,
    }


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_vllm_gumbel_sample(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_vllm_gumbel_sample(**kwargs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.dtype == ref_out.dtype == torch.int64
    assert res_out.shape == ref_out.shape
    assert torch.equal(res_out, ref_out)


def compute_ref_performance(**kwargs):
    return measure_call(compute_ref, kwargs, kwargs["logits"].device)


def compute_res_performance(**kwargs):
    return measure_call(compute_res, kwargs, kwargs["logits"].device)


@label("v1_vllm_gumbel_sample")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_gumbel_sample(config):
    (rows, vocab, dtype, apply_temperature, is_drafting, pattern) = config[0]
    inputs = input_build(config)
    device = kernelgenbench.device
    ref_inputs = move(inputs, device)
    res_inputs = move(inputs, device)
    initial = {k: v.clone() for k, v in res_inputs.items() if isinstance(v, torch.Tensor)}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    assert res_out.shape == (rows,)
    assert_unchanged(
        {k: v for k, v in res_inputs.items() if isinstance(v, torch.Tensor)}, initial
    )

    # Temperature 0 (including the -1 invalid rows, whose temperature is forced
    # to 0) must reduce to a plain argmax over the row.
    mapping = res_inputs["expanded_idx_mapping"]
    temp = res_inputs["temperature"]
    row_temp = torch.where(mapping >= 0, temp[mapping.long().clamp_min(0)], torch.zeros_like(temp[mapping.long().clamp_min(0)]))
    plain = torch.argmax(res_inputs["logits"].float(), dim=-1)
    zero_rows = torch.where(row_temp == 0)[0]
    assert torch.equal(res_out[zero_rows], plain[zero_rows])
    if is_drafting:
        # Draft rows salt the noise position, so the same seed/pos draws a
        # different token than the target stream.
        target = compute_res(**{**res_inputs, "is_drafting": False})
        assert not torch.equal(target, res_out)

    perf = move(inputs, device)
    ref_time = compute_ref_performance(**perf)
    res_time = compute_res_performance(**perf)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
