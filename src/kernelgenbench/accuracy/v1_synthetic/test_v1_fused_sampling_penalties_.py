"""Accuracy and performance for v1_fused_sampling_penalties_.

Six draft-expanded rows cover three requests. Request 2 keeps the neutral
parameters (repetition 1.0, frequency 0, presence 0), so its last row must stay
bit-identical. The other rows combine a prompt-bit hit, committed output counts
and draft-token counts from ``expanded_local_pos``.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.sampling_test_utils import (
    assert_unchanged,
    measure_inplace,
    move,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

VOCABS = [13, 129]
COMBOS = [
    (vocab, dtype, row_padding)
    for vocab in VOCABS
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for row_padding in (False, True)
]


def input_build(config):
    vocab, dtype, row_padding = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    rows = 6
    base = torch.randn((rows, vocab + (8 if row_padding else 0)), generator=generator).to(dtype)
    logits = base[:, :vocab]
    counts = torch.randint(0, 3, (3, vocab), generator=generator).to(torch.int32)
    packed = torch.randint(
        -(2**31), 2**31 - 1, (3, (vocab + 31) // 32), generator=generator
    ).to(torch.int32)
    return {
        "logits": logits,
        "expanded_idx_mapping": torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64),
        # Row 5 (request 2) uses the neutral penalty parameters.
        "token_ids": torch.tensor([1, 4, 4, 3, 7, 0], dtype=torch.int64),
        "expanded_local_pos": torch.tensor([0, 1, 2, 0, 1, 0], dtype=torch.int32),
        "repetition_penalty": torch.tensor([1.4, 0.7, 1.0], dtype=torch.float32),
        "frequency_penalty": torch.tensor([0.2, 0.6, 0.0], dtype=torch.float32),
        "presence_penalty": torch.tensor([0.4, 0.1, 0.0], dtype=torch.float32),
        "prompt_bin_mask": packed,
        "output_bin_counts": counts,
    }


def compute_ref(logits, **kwargs):
    return kernelgenbench.baseline.v1_fused_sampling_penalties_(logits, **kwargs)


def compute_res(logits, **kwargs):
    return kernelgenbench.solution.v1_fused_sampling_penalties_(logits, **kwargs)


def compute_ref_performance(initial, logits, **kwargs):
    return measure_inplace(compute_ref, {"logits": logits, **kwargs}, initial, logits.device)


def compute_res_performance(initial, logits, **kwargs):
    return measure_inplace(compute_res, {"logits": logits, **kwargs}, initial, logits.device)


@label("v1_fused_sampling_penalties_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_fused_sampling_penalties_(config):
    vocab, dtype, row_padding = config
    inputs = input_build(config)
    device = kernelgenbench.device
    ref_inputs = move(inputs, device)
    res_inputs = move(inputs, device)
    initial = res_inputs["logits"].clone()
    metadata = {
        k: v.clone()
        for k, v in res_inputs.items()
        if isinstance(v, torch.Tensor) and k != "logits"
    }

    assert compute_ref(**ref_inputs) is None
    assert compute_res(**res_inputs) is None
    out = res_inputs["logits"]
    assert out.stride(0) == vocab + (8 if row_padding else 0)
    assert_unchanged(
        {k: v for k, v in res_inputs.items() if k != "logits"}, metadata
    )

    rtol, atol = {
        torch.float32: (1e-6, 1e-6),
        torch.float16: (2e-3, 1e-3),
        torch.bfloat16: (1e-2, 8e-3),
    }[dtype]
    torch.testing.assert_close(out, ref_inputs["logits"], rtol=rtol, atol=atol)

    # Neutral request 2 row is untouched.
    assert torch.equal(out[5], initial[5])
    # Penalised rows moved.
    assert not torch.equal(out[:5], initial[:5])
    # The neutral row's row-5 token 0 draft history came from row 5 only, and
    # frequency/presence are zero, so even the shared-vocabulary entries match.
    assert torch.equal(out[5], initial[5])

    perf_ref = move(inputs, device)
    perf_res = move(inputs, device)
    ref_time = compute_ref_performance(perf_ref["logits"].clone(), **perf_ref)
    res_time = compute_res_performance(perf_res["logits"].clone(), **perf_res)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
