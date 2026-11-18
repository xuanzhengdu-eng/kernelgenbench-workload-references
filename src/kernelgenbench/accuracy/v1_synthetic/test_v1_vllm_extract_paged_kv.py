"""Accuracy and performance for v1_vllm_extract_paged_kv.

Operator: vLLM ``extract_from_kv_cache`` (hidden-state transfer helper).
Official source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py#L37-L46

Contract: ``kv_cache`` is ``[blocks, heads, page, width]``; each entry of
``slot_mapping`` addresses ``slot // page`` and ``slot % page``, and the result
is truncated to ``num_tokens`` rows.  The result is a fresh tensor with the
cache dtype (never an alias), may be empty, may repeat slots, and the cache is
never modified.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows502_511_checks as checks


# (blocks, heads, page size, width, slot_mapping, num_tokens, dtype, strided cache)
# Strided caches keep an even inner stride by slicing a doubled last axis.
COMBOS = [
    (3, 2, 4, 5, (0, 5, 11, 5), 3, torch.float32, False),
    (2, 3, 2, 8, (3, 3, 0), 3, torch.bfloat16, True),
    (2, 1, 4, 3, (), 0, torch.float16, False),
    (2, 2, 4, 4, (1, 6), 0, torch.float32, True),
    (4, 2, 2, 6, (7, 0, 3, 7, 1), 5, torch.float16, False),
    (1, 4, 8, 4, (0, 7, 2, 5, 1, 6, 3, 4), 8, torch.bfloat16, False),
    (3, 1, 4, 6, (9, 2, 10), 2, torch.float32, False),
]


def input_build(config):
    """Build a page-major cache on CPU.

    ``kv_cache`` is (blocks, heads, page, width); ``strided`` builds it from a
    doubled last axis, so the width stride is 2 and the tensor is not
    contiguous.  ``slot_mapping`` is int64 and ``num_tokens`` truncates rows.
    """
    blocks, heads, page, width, slots, num_tokens, dtype, strided = config
    raw_width = width * (2 if strided else 1)
    raw = torch.arange(blocks * heads * page * raw_width, dtype=torch.float32)
    raw = raw.reshape(blocks, heads, page, raw_width).to(dtype)
    kv_cache = raw[..., ::2] if strided else raw
    return {
        "kv_cache": kv_cache,
        "slot_mapping": torch.tensor(slots, dtype=torch.int64),
        "num_tokens": num_tokens,
    }


def compute_ref(kv_cache, slot_mapping, num_tokens):
    return kernelgenbench.baseline.v1_vllm_extract_paged_kv(
        kv_cache, slot_mapping, num_tokens)


def compute_res(kv_cache, slot_mapping, num_tokens):
    return kernelgenbench.solution.v1_vllm_extract_paged_kv(
        kv_cache, slot_mapping, num_tokens)


def compare(ref_out, res_out, kv_cache, expected_rows):
    assert res_out.shape == (expected_rows, kv_cache.shape[1], kv_cache.shape[3])
    assert res_out.dtype == kv_cache.dtype and ref_out.dtype == kv_cache.dtype
    assert torch.equal(res_out, ref_out)
    assert not torch._C._is_alias_of(res_out, kv_cache)


def compute_ref_performance(**inputs):
    return checks.measure(compute_ref, inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(compute_res, inputs, kernelgenbench.device)


@label("v1_vllm_extract_paged_kv")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_extract_paged_kv(config):
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    cache_before = res_inputs["kv_cache"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    if torch.device(kernelgenbench.device).type == "cuda":
        torch.cuda.synchronize()

    rows = min(config[5], len(config[4]))
    compare(ref_out, res_out, res_inputs["kv_cache"], rows)
    assert torch.equal(res_inputs["kv_cache"], cache_before)

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
