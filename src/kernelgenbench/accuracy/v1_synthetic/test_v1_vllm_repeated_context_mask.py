"""Accuracy and performance for v1_vllm_repeated_context_mask.

Pinned entry ``repeated_context_mask`` (GPU Triton ``_repeated_context_mask_kernel``):
per row, report whether the context window already occurred in that request's
history (optionally limited to the most recent ``max_history`` positions and
optionally including the prompt), whether an incomplete context should be
skipped, and whether an earlier context of the same speculative block matches.
Padding rows (``req_index < 0``) are always reported as not repeated.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.spec_decode_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (max_history, include_prompt, skip_partial_context, history offsets, speculative locals)
CASES = [
    (None, False, False, False, False),
    (3, False, False, True, False),
    (2, True, False, True, False),
    (None, False, True, False, False),
    (3, False, False, False, True),
    (2, True, True, True, True),
]
COMBOS = [(case, dtype, device) for case in CASES
          for dtype in (torch.int32, torch.int64) for device in ("cuda", "cpu")]
WIDTH = 3


def input_build(config):
    case, dtype, _device = config
    limit, include, skip, use_offsets, local = case
    return {
        "all_token_ids": torch.tensor([[1, 2, 1, 2, 1, 3, 0], [7, 8, 9, 9, 8, 9, 0]],
                                      dtype=dtype),
        "req_indices": torch.tensor([0, 0, 1, -1, 1], dtype=torch.int64),
        "prompt_lens": torch.tensor([2, 1], dtype=torch.int64),
        "total_lens": torch.tensor([6, 6], dtype=torch.int64),
        "contexts": torch.tensor([[1, 2, 1], [1, 2, 1], [7, 8, 9], [-1, -1, -1], [8, 9, 9]],
                                 dtype=dtype),
        "max_history": limit,
        "include_prompt": include,
        "skip_partial_context": skip,
        "history_offsets": torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64) if use_offsets else None,
        "local_positions": torch.tensor([0, 1, 0, 0, 1], dtype=torch.int64) if local else None,
        "num_speculative_steps": 2 if local else 0,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_repeated_context_mask(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_repeated_context_mask(**inputs)


def expected_mask(case, cpu):
    """Independent scan of the documented window/local-context contract.

    The CPU upstream branch reports padded rows with negative context as
    repeated (``contexts < 0``), while the Triton branch returns early on
    request index -1.
    """
    limit, include, skip, use_offsets, local = case
    all_ids = [[1, 2, 1, 2, 1, 3, 0], [7, 8, 9, 9, 8, 9, 0]]
    req = [0, 0, 1, -1, 1]
    prompt = [2, 1]
    total = [6, 6]
    contexts = [[1, 2, 1], [1, 2, 1], [7, 8, 9], [-1, -1, -1], [8, 9, 9]]
    offsets = [0, 1, 2, 0, 1] if use_offsets else [0] * 5
    locals_ = [0, 1, 0, 0, 1] if local else [0] * 5
    result = []
    for row in range(5):
        request = req[row]
        repeated = False
        if request >= 0:
            if skip and total[request] - prompt[request] < WIDTH:
                repeated = True
            else:
                start = 0 if include else prompt[request]
                history = all_ids[request][start:total[request]]
                floor = 0 if limit is None else max(0, len(history) - max(0, limit - offsets[row]))
                for pos in range(floor, len(history)):
                    preceding = ([-1] * WIDTH + history[:pos])[-WIDTH:]
                    if preceding == contexts[row]:
                        repeated = True
                        break
            if local:
                maximum = 2 if limit is None else min(2, limit)
                for off in range(1, maximum + 1):
                    previous = row - off
                    if (previous >= 0 and locals_[row] >= off
                            and req[previous] == request
                            and locals_[row] == locals_[previous] + off
                            and contexts[previous] == contexts[row]):
                        repeated = True
                        break
                if include and any(token < 0 for token in contexts[row]):
                    repeated = True
        elif cpu and local and include and any(token < 0 for token in contexts[row]):
            repeated = True
        result.append(repeated)
    return result


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.dtype == torch.bool and ref_out.dtype == torch.bool
    assert res_out.shape == ref_out.shape
    assert torch.equal(res_out, ref_out)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs, device=inputs["contexts"].device)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs, device=inputs["contexts"].device)


@label("v1_vllm_repeated_context_mask")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_repeated_context_mask(config):
    case, dtype, device_name = config
    device = torch.device(device_name)
    inputs = input_build(config)
    ref_inputs = fresh(inputs, device)
    res_inputs = fresh(inputs, device)
    assert ref_inputs["all_token_ids"].shape == (2, 7)
    assert ref_inputs["contexts"].shape == (5, WIDTH)
    assert ref_inputs["contexts"].dtype == dtype
    assert ref_inputs["req_indices"].shape == (5,)
    if case[4]:
        assert ref_inputs["local_positions"].shape == (5,)
        assert ref_inputs["num_speculative_steps"] == 2

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    expected = expected_mask(case, cpu=(device_name == "cpu"))
    assert res_out.tolist() == expected, "repeated-context mask differs from the documented scan"
    # Padding rows are never repeated on the Triton path; the CPU branch marks
    # a padded row whose context still contains start padding.
    if device_name == "cuda":
        assert not bool(res_out[3]), "padding row must be reported as not repeated"
    else:
        assert bool(res_out[3]) == expected[3]
    for name in ("all_token_ids", "req_indices", "contexts"):
        assert torch.equal(ref_inputs[name], inputs[name].to(ref_inputs[name].device))

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
