"""Accuracy and performance for v1_sgl_apply_token_bitmask_ (Triton backend).

Official source (pinned commit 62384de1ae5e3bbd3159afd1639525b0bf39fbf9):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/grammar/bitmask_ops.py#L84-L141

Contract: ``logits`` is a contiguous CUDA 1-D/2-D float32/float16/bfloat16
tensor; ``bitmask`` is contiguous int32 with one bit per token, bit 1 = allowed.
Each selected row is masked in place: disallowed columns in
``[0, min(vocab, words*32))`` become ``-inf``, the tail beyond the mask width is
untouched, ``indices`` (int32) chooses rows and is not modified.  Returns
``None``.

Item: v1_sgl_apply_token_bitmask_ (723-row 584), Triton backend.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows582_591_checks as checks


# rows, vocab, mask words, dtype, selected rows (None = all), 1-D input.
CASES = [
    (1, 35, 2, torch.float32, None, True),
    (3, 64, 2, torch.float16, (2, 0), False),
    (4, 77, 2, torch.bfloat16, (1, 3), False),
    (2, 128, 4, torch.float32, None, False),
]
COMBOS = [(case,) for case in CASES]


def input_build(config):
    """logits ``[rows, vocab]`` or ``[vocab]``; bitmask ``[rows, words]`` int32.

    The values cycle through -20..20 so every masked position is finite before
    the call.  The four mask patterns cover all-zero, mixed, all-one and the
    sign-bit-only word.
    """
    (rows, vocab, words, dtype, chosen, one), = config
    logits = (torch.arange(rows * vocab, dtype=torch.float32).reshape(rows, vocab) % 41 - 20).to(dtype=dtype)
    bitmask = torch.empty((rows, words), dtype=torch.int32)
    pattern = [0x55555555, 0x01234567, 0x7FFFFFFF, -2147483648]
    for row in range(rows):
        for word in range(words):
            bitmask[row, word] = pattern[(row + word) % 4]
    indices = torch.tensor(chosen, dtype=torch.int32) if chosen is not None else None
    return {
        "logits": logits[0] if one else logits,
        "bitmask": bitmask[0] if one else bitmask,
        "indices": indices,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_apply_token_bitmask_triton(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_apply_token_bitmask_triton(**inputs)


def compare(ref_out, res_out):
    assert ref_out is None and res_out is None


def compute_ref_performance(logits, bitmask, indices):
    initial = logits.clone()

    def call():
        logits.copy_(initial)
        return compute_ref(logits=logits, bitmask=bitmask, indices=indices)

    return checks.measure(call, {}, logits.device)


def compute_res_performance(logits, bitmask, indices):
    initial = logits.clone()

    def call():
        logits.copy_(initial)
        return compute_res(logits=logits, bitmask=bitmask, indices=indices)

    return checks.measure(call, {}, logits.device)


@label("v1_sgl_apply_token_bitmask_triton")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_apply_token_bitmask_triton(config):
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    initial = ref_inputs["logits"].clone()
    bitmask_before = ref_inputs["bitmask"].clone()
    indices_before = None if ref_inputs["indices"] is None else ref_inputs["indices"].clone()
    ref_ptr = ref_inputs["logits"].data_ptr()
    res_ptr = res_inputs["logits"].data_ptr()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)
    torch.testing.assert_close(ref_inputs["logits"], res_inputs["logits"], rtol=0, atol=0)

    # In-place mutation, same storage, and read-only bitmask/indices.
    assert ref_inputs["logits"].data_ptr() == ref_ptr
    assert res_inputs["logits"].data_ptr() == res_ptr
    torch.testing.assert_close(ref_inputs["bitmask"], bitmask_before, rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["bitmask"], bitmask_before, rtol=0, atol=0)
    if indices_before is not None:
        torch.testing.assert_close(ref_inputs["indices"], indices_before, rtol=0, atol=0)
        torch.testing.assert_close(res_inputs["indices"], indices_before, rtol=0, atol=0)

    # Recompute the packed-bit contract independently.
    logits2d = initial.view(1, -1) if initial.ndim == 1 else initial
    mask2d = bitmask_before.view(1, -1) if bitmask_before.ndim == 1 else bitmask_before
    width = min(logits2d.shape[1], mask2d.shape[1] * 32)
    rows = range(logits2d.shape[0]) if inputs["indices"] is None else [int(i) for i in inputs["indices"]]
    expected = initial.clone()
    expected2d = expected.view(1, -1) if expected.ndim == 1 else expected
    columns = torch.arange(width, device=expected.device)
    for row in rows:
        allowed = ((mask2d[row, columns // 32].to(torch.int64) >> (columns % 32)) & 1).bool()
        expected2d[row, :width].masked_fill_(~allowed, float("-inf"))
    torch.testing.assert_close(res_inputs["logits"], expected, rtol=0, atol=0)
    # Unselected rows keep every original value.
    touched = set(rows)
    res2d = res_inputs["logits"].view(1, -1) if res_inputs["logits"].ndim == 1 else res_inputs["logits"]
    initial2d = initial.view(1, -1) if initial.ndim == 1 else initial
    for row in range(logits2d.shape[0]):
        if row not in touched:
            torch.testing.assert_close(res2d[row], initial2d[row], rtol=0, atol=0)

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return checks.result(ref_time, res_time)
