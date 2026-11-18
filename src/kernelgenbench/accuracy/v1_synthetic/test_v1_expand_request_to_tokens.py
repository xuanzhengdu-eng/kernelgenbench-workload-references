"""Accuracy and performance for v1_expand_request_to_tokens.

Operator: vLLM ``expand_batch_to_tokens`` (one scalar per request -> one scalar
per speculative token; Triton ``expand_kernel``).
Official source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/rejection_sampler.py#L648-L687
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/rejection_sampler.py#L1023-L1067

Contract: ``cu_num_tokens`` is the inclusive cumulative token count per request;
request ``i`` owns ``cu[i] - cu[i-1]`` output elements.  Each value equal to
``replace_from`` becomes ``replace_to`` before expansion.  The output has length
``num_tokens``, the same dtype/device as ``x``, and neither input is modified.
The upstream Triton kernel specializes ``MAX_NUM_TOKENS = MAX_SPEC_LEN = 128``,
so per-request counts are at most 128 (the legal boundary is covered).

Item: v1_expand_request_to_tokens (723-row 641).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows632_641_test_utils import (
    fresh, measure, timings)

# Per-request token counts (inclusive cumulative is built from them); 128 is the
# upstream MAX_SPEC_LEN boundary, 0 exercises empty requests and an all-empty batch.
COUNTS = [(0, 1, 3), (2, 0, 1), (128, 0, 4), (0, 0, 0)]
COMBOS = [(counts, dtype, replace) for counts in COUNTS
          for dtype in (torch.float32, torch.int32, torch.int64)
          for replace in (False, True)]


def input_build(config):
    """x: [3] values [0, 2, -1]; cu_num_tokens: [3] int32 inclusive cumulative;
    num_tokens: sum of the counts.
    """
    counts, dtype, replace = config
    cu_num_tokens = torch.tensor(counts, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return {
        "x": torch.tensor([0, 2, -1], dtype=dtype),
        "cu_num_tokens": cu_num_tokens,
        "num_tokens": sum(counts),
        "replace_from": 0,
        "replace_to": 3 if replace else 0,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_expand_request_to_tokens(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_expand_request_to_tokens(**inputs)


def compare(ref_out, res_out, ref_inputs, res_inputs):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape, "output length must equal num_tokens"
    assert res_out.shape[0] == res_inputs["num_tokens"]
    assert res_out.dtype == ref_out.dtype == res_inputs["x"].dtype
    assert res_out.device.type == ref_out.device.type
    assert torch.equal(res_out.cpu(), ref_out.cpu()), "expanded values differ"
    # The value remap must be applied before the per-request repeat.
    expected = ref_inputs["x"].cpu().clone()
    if ref_inputs["replace_to"] != ref_inputs["replace_from"]:
        expected = torch.where(expected == ref_inputs["replace_from"],
                               torch.tensor(ref_inputs["replace_to"], dtype=expected.dtype),
                               expected)
    counts = torch.diff(torch.cat((torch.zeros(1, dtype=torch.int32),
                                   ref_inputs["cu_num_tokens"].cpu())))
    assert torch.equal(res_out.cpu(), torch.repeat_interleave(expected, counts.long()))


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs, kernelgenbench.device)


@label("v1_expand_request_to_tokens")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_expand_request_to_tokens(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    x_before = inputs["x"].clone()
    cu_before = inputs["cu_num_tokens"].clone()

    compare(compute_ref(**ref_inputs), compute_res(**res_inputs),
            ref_inputs, res_inputs)

    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["x"].cpu(), x_before, rtol=0, atol=0)
        torch.testing.assert_close(side["cu_num_tokens"].cpu(), cu_before, rtol=0, atol=0)
    return timings(compute_ref_performance(**inputs),
                   compute_res_performance(**inputs))
