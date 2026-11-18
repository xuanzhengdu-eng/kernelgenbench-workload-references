"""Accuracy and performance for v1_vllm_no_repeat_ngram.

For every row the history of its request slot is scanned for occurrences of the
current (n-1)-token prefix; the following token is set to -inf in place.  The
test recomputes the blocked set independently and also checks that the history,
slot and position inputs are read-only.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows612_621_test_utils as utils
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# ngram size per request; request slot and current position per logits row
COMBOS = [((0, 0, 0), (0, 1, 2), (4, 4, 4)),
          ((1, 2, 3), (0, 1, 2), (5, 5, 5)),
          ((2, 3, 1), (2, 0, 1, 0), (5, 4, 5, -1)),
          ((12, 0, 2), (0, 1, 2), (5, 2, 5)),
          ((3, 2, 1), (0, 0, 1, 2), (2, 5, 4, 5)),
          ((1, 1, 1), (0, 1, 2), (5, 5, 5))]
VOCAB = 7
HISTORY = [[1, 2, 1, 2, 1, 3], [4, 1, 4, 1, 4, 2], [2, 9, -1, 2, 9, 4]]


def input_build(config):
    sizes, slots, positions = config
    return {
        "logits": torch.arange(len(slots) * VOCAB, dtype=torch.float32).reshape(
            len(slots), VOCAB) * 0.03125,
        "history": torch.tensor(HISTORY, dtype=torch.int32),
        "request_slots": torch.tensor(slots, dtype=torch.int64),
        "positions": torch.tensor(positions, dtype=torch.int64),
        "ngram_sizes": torch.tensor(sizes, dtype=torch.int32),
    }


def compute_ref(logits, history, request_slots, positions, ngram_sizes):
    return kernelgenbench.baseline.v1_vllm_no_repeat_ngram(
        logits, history, request_slots, positions, ngram_sizes)


def compute_res(logits, history, request_slots, positions, ngram_sizes):
    return kernelgenbench.solution.v1_vllm_no_repeat_ngram(
        logits, history, request_slots, positions, ngram_sizes)


def _expected(config, device):
    sizes, slots, positions = config
    expected = torch.arange(len(slots) * VOCAB, dtype=torch.float32).reshape(
        len(slots), VOCAB) * 0.03125
    for row, (slot, position) in enumerate(zip(slots, positions)):
        size = sizes[slot]
        length = position + 1
        if size == 0 or size > len(HISTORY[0]) or size > length:
            continue
        sequence = HISTORY[slot][:length]
        tail = sequence[length - (size - 1):] if size > 1 else []
        for start in range(length - size + 1):
            if size == 1 or sequence[start:start + size - 1] == tail:
                token = sequence[start + size - 1]
                if 0 <= token < VOCAB:
                    expected[row, token] = float("-inf")
    return expected.to(device)


def compare(ref_out, res_out, config, ref_inputs, res_inputs):
    expected = _expected(config, ref_out.device)
    assert ref_out is ref_inputs["logits"]
    assert res_out is res_inputs["logits"]
    torch.testing.assert_close(ref_out, expected, rtol=0, atol=0)
    torch.testing.assert_close(res_out, expected, rtol=0, atol=0)


def _timing(config, operation):
    base = input_build(config)
    device = kernelgenbench.device
    return utils.event_time(lambda inputs: operation(**inputs),
                            lambda: utils.fresh(base, device),
                            warmup=10, repetitions=50)


def compute_ref_performance(logits, history, request_slots, positions, ngram_sizes):
    return _timing((tuple(ngram_sizes.tolist()), tuple(request_slots.tolist()),
                    tuple(positions.tolist())), compute_ref)


def compute_res_performance(logits, history, request_slots, positions, ngram_sizes):
    return _timing((tuple(ngram_sizes.tolist()), tuple(request_slots.tolist()),
                    tuple(positions.tolist())), compute_res)


@label("v1_vllm_no_repeat_ngram")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_no_repeat_ngram(config):
    inputs = input_build(config)
    ref_inputs = utils.fresh(inputs, kernelgenbench.device)
    res_inputs = utils.fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, ref_inputs, res_inputs)
    for name, side in (("reference", ref_inputs), ("solution", res_inputs)):
        for key in ("history", "request_slots", "positions", "ngram_sizes"):
            utils.assert_unchanged({key: inputs[key]}, side)
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
