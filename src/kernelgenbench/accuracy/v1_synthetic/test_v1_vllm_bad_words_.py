"""Accuracy and performance for v1_vllm_bad_words_.

Inputs are four draft-expanded rows over two requests. Row history combines the
committed tokens after the prompt with the speculative ``input_ids`` window, so
the same bad-word list masks different tokens in the committed and speculative
cases. The expected masked positions below are derived by hand from that rule.
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

# mode, row-strided logits.
CASES = [
    ("committed", False),
    ("committed", True),
    ("speculative", False),
    ("speculative", True),
    ("none", False),
]
COMBOS = [
    (mode, strided, dtype)
    for mode, strided in CASES
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]

# (row, token) pairs the kernel must mask with -inf for each history mode.
EXPECTED_MASKED = {
    # [5,6], [4,5,7] match the committed history; [9] has an empty prefix and
    # therefore always masks token 9.
    "committed": ((0, 6), (0, 7), (0, 9), (1, 9), (2, 9), (3, 11), (3, 12)),
    # The speculative window turns [6,13] into a hit on row 1 and [6,7,14] on
    # row 2; [9] and the single-token word [12] still mask unconditionally.
    "speculative": ((0, 9), (1, 9), (1, 13), (2, 9), (2, 14), (3, 11), (3, 12)),
    "none": (),
}


def input_build(config):
    mode, strided, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    # Row stride is 24 while the vocabulary is 16, so the kernel must honour
    # logits.stride(0) instead of assuming a dense row.
    base = torch.randn((4, 24 if strided else 16), dtype=dtype, generator=generator)
    logits = base[:, :16]
    words = [5, 6, 4, 5, 7, 9] if mode != "speculative" else [6, 13, 6, 7, 14, 9]
    counts = [3, 2] if mode != "none" else [0, 0]
    return {
        "logits": logits,
        "expanded_idx_mapping": torch.tensor([0, 0, 0, 1], dtype=torch.int32),
        "bad_word_token_ids": torch.tensor(
            [words, [10, 11, 12, 0, 0, 0]], dtype=torch.int32
        ),
        "bad_word_offsets": torch.tensor(
            [[0, 2, 5, 6], [0, 2, 3, 3]], dtype=torch.int32
        ),
        "num_bad_words": torch.tensor(counts, dtype=torch.int32),
        "all_token_ids": torch.tensor(
            [[90, 91, 3, 4, 5, 0, 0, 0], [80, 10, 10, 0, 0, 0, 0, 0]],
            dtype=torch.int32,
        ),
        "prompt_len": torch.tensor([2, 1], dtype=torch.int32),
        "total_len": torch.tensor([5, 3], dtype=torch.int32),
        "input_ids": torch.tensor([5, 6, 7, 10], dtype=torch.int32),
        "expanded_local_pos": torch.tensor([0, 1, 2, 0], dtype=torch.int32),
    }


def compute_ref(logits, **metadata):
    return kernelgenbench.baseline.v1_vllm_bad_words_(logits, **metadata)


def compute_res(logits, **metadata):
    return kernelgenbench.solution.v1_vllm_bad_words_(logits, **metadata)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0, equal_nan=True)


def compute_ref_performance(initial, logits, **metadata):
    return measure_inplace(
        compute_ref, {"logits": logits, **metadata}, initial, logits.device
    )


def compute_res_performance(initial, logits, **metadata):
    return measure_inplace(
        compute_res, {"logits": logits, **metadata}, initial, logits.device
    )


@label("v1_vllm_bad_words_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_bad_words_(config):
    mode, strided, dtype = config
    inputs = input_build(config)
    device = kernelgenbench.device
    ref_inputs = move(inputs, device)
    res_inputs = move(inputs, device)
    initial = res_inputs["logits"].clone()
    metadata = {k: v.clone() for k, v in res_inputs.items() if k != "logits"}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)

    # Both entries rewrite the caller's tensor and return that same object.
    assert ref_out is ref_inputs["logits"]
    assert res_out is res_inputs["logits"]
    compare(ref_out, res_out, dtype)
    assert res_inputs["logits"].stride(0) == (24 if strided else 16)

    # Only the expected bad-word completions become -inf; every other entry
    # keeps the exact input value.
    rows, tokens = torch.where(torch.isneginf(res_out))
    masked = set(zip(rows.tolist(), tokens.tolist()))
    assert masked == set(EXPECTED_MASKED[mode]), (mode, sorted(masked))
    keep = torch.ones_like(initial, dtype=torch.bool)
    if EXPECTED_MASKED[mode]:
        rows, tokens = zip(*EXPECTED_MASKED[mode])
        keep[list(rows), list(tokens)] = False
    assert torch.equal(res_out[keep], initial[keep])
    assert_unchanged({k: v for k, v in res_inputs.items() if k != "logits"}, metadata)

    perf_ref = move(inputs, device)
    perf_res = move(inputs, device)
    ref_time = compute_ref_performance(perf_ref["logits"].clone(), **perf_ref)
    res_time = compute_res_performance(perf_res["logits"].clone(), **perf_res)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
