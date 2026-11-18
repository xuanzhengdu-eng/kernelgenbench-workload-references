"""Accuracy and performance for v1_vllm_topk_scores.

Operator: vLLM ``compute_topk_scores`` (sampled-token score, top-k ids and rank).
Official source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/logprob.py#L84-L108
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/logprob.py#L111-L185

Contract: returns ``LogprobsTensors(logprob_token_ids, logprobs,
selected_token_ranks, cu_num_generated_tokens, cu_num_generated_tokens_tensor)``.
Column 0 is always the sampled token; the next ``num_logprobs`` columns are the
top-k ids, or the per-request custom ids when ``max_per_req_token_ids > 0``
(duplicates kept, invalid columns masked to -inf).  ``logits_mode`` returns raw
FP32 logits instead of log-softmax scores.  The rank counts every score
``>=`` the sampled score, so ties all count.  ``cu_num_logits`` passes through
unchanged: a list goes to field 3, a tensor to field 4.  Logits are read-only.

Item: v1_vllm_topk_scores (723-row 633).
"""
import types

import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows632_641_test_utils import (
    fresh, measure, timings)

# num_top, scores-as-logits, cumulative metadata type, add ties, padded rows,
# optional per-request custom ids.
CASES = [
    (0, False, "none", False, False),
    (2, False, "list", True, False),
    (3, True, "tensor", False, True),
    (1, True, "none", True, True),
    (2, False, "tensor", False, False, "custom"),
    (0, True, "none", False, True, "custom"),
]

COMBOS = [(case, dtype) for case in CASES
          for dtype in (torch.float32, torch.float16, torch.bfloat16)]


def input_build(config):
    """logits: [3, 8] (stride 16 in the padded case); sampled ids: [3] int64;
    optional boundaries [0, 1, 3] as list/tensor; custom state has counts
    [2, 0] and token ids [[7, 2, 0], [0, 0, 0]] with mapping [0, 1, 0].
    """
    case, dtype = config
    k, as_logits, cu, tie, pad = case[:5]
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    logits = torch.randn((3, 8), dtype=dtype, generator=generator)
    if tie:
        logits[0, 0] = logits[0, 1] = logits[0, 2] = 2.0
    if pad:
        base = torch.empty((3, 16), dtype=dtype)
        base[:, :8] = logits
        logits = base[:, :8]
    boundaries = None if cu == "none" else (
        [0, 1, 3] if cu == "list" else torch.tensor([0, 1, 3], dtype=torch.int32))
    inputs = {
        "logits": logits,
        "num_logprobs": k,
        "sampled_token_ids": torch.tensor([0, 3, 5], dtype=torch.int64),
        "cu_num_logits": boundaries,
        "logits_mode": as_logits,
    }
    if len(case) > 5:
        counts = torch.tensor([2, 0], dtype=torch.int32)
        tokens = torch.tensor([[7, 2, 0], [0, 0, 0]], dtype=torch.int32)
        inputs.update(
            logprob_token_ids_state=types.SimpleNamespace(
                num_token_ids=types.SimpleNamespace(gpu=counts),
                token_ids=types.SimpleNamespace(gpu=tokens)),
            expanded_idx_mapping=torch.tensor([0, 1, 0], dtype=torch.int32),
            max_per_req_token_ids=3)
    return inputs


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_topk_scores(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_topk_scores(**inputs)


def compare(ref_out, res_out, dtype, ref_inputs, res_inputs):
    assert isinstance(ref_out, tuple) and len(ref_out) == 5
    assert isinstance(res_out, tuple) and len(res_out) == 5
    for name, index in (("logprob_token_ids", 0), ("logprobs", 1),
                        ("selected_token_ranks", 2)):
        left, right = ref_out[index], res_out[index]
        assert isinstance(right, torch.Tensor), f"{name}: not a tensor"
        assert right.shape == left.shape, f"{name}: shape"
        assert right.dtype == left.dtype, f"{name}: dtype"
        if left.is_floating_point():
            rtol, atol = {torch.float32: (2e-5, 2e-5),
                          torch.float16: (2e-3, 1e-3),
                          torch.bfloat16: (1e-2, 8e-3)}[dtype]
            torch.testing.assert_close(right.float(), left.float(), rtol=rtol,
                                       atol=atol, equal_nan=True)
        else:
            assert torch.equal(right, left), f"{name}: ids differ"
    # Cumulative boundaries pass through untouched: list -> field 3, tensor -> field 4.
    boundaries = ref_inputs["cu_num_logits"]
    if isinstance(boundaries, torch.Tensor):
        assert ref_out[3] is None and res_out[3] is None
        assert res_out[4] is res_inputs["cu_num_logits"]
        assert torch.equal(res_out[4].cpu(), boundaries.cpu())
    else:
        assert ref_out[4] is None and res_out[4] is None
        assert res_out[3] is res_inputs["cu_num_logits"]
        assert res_out[3] == boundaries


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs, kernelgenbench.device)


@label("v1_vllm_topk_scores")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_topk_scores(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    original_logits = inputs["logits"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config[1], ref_inputs, res_inputs)

    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["logits"].cpu(), original_logits,
                                   rtol=0, atol=0)
    return timings(compute_ref_performance(**inputs),
                   compute_res_performance(**inputs))
