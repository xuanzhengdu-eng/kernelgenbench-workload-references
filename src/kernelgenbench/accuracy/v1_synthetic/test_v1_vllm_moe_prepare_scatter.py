"""Accuracy and performance for v1_vllm_moe_prepare_scatter (vLLM CUDA MoE).

Official entry: vLLM ``torch.ops._moe_C.moe_prepare_scatter``, pinned commit
f358f0578a83f4e1fe6fc49a492d3ae9362de8e5, C++ lines 200-214 and kernel lines
130-182.  The candidate binding imports the pinned stable-libtorch extension
and calls that dispatcher; the Python wrapper's int32 cast and 2-D inverse
view are reproduced in the binding.

Inputs: ``topk_ids`` [tokens, top_k] int32/int64, ``token_expert_indices`` same
shape int32, optional int32 ``expert_map`` [n_expert], int64
``expert_first_token_offset`` [n_local_expert+1], int32 scratch of numel()
entries (plus a tail guard).  Outputs are the offset tensor (same object) and
an int32 inverse view aliasing ``scatter_idx``.  Invalid and non-local IDs sort
to n_expert / id+n_expert; the sort is stable.  Inputs are not modified.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows202_211_test_utils import fresh, measure, result


# rows, top_k, n_expert, n_local_expert, expert_map, id dtype, source permutation
COMBOS = [
    (((3, 0), (1, 2), (3, 0)), 2, 4, 4, None, torch.int32, None),
    (((4, 2), (0, 1), (-1, 8), (3, 4)), 2, 5, 3, (2, -1, 0, -1, 1), torch.int64, (7, 2, 0, 5, 1, 4, 6, 3)),
    (((-1, 2), (7, -5), (2, 1)), 2, 3, 1, (-1, 0, -1), torch.int32, None),
    ((), 2, 4, 4, None, torch.int32, None),
    (((0, 0, 0), (0, 0, 0), (0, 0, 0)), 3, 1, 1, None, torch.int32, None),
    (((0, 1, 2), (3, 4, 5)), 3, 6, 6, None, torch.int64, (5, 4, 3, 2, 1, 0)),
    (((2, 2, 0), (0, 1, 1)), 3, 3, 2, (1, -1, 0), torch.int32, None),
]


def input_build(config):
    rows, top_k, n_expert, n_local, mapping, dtype, perm = config
    n = len(rows) * top_k
    topk_ids = torch.tensor(rows, dtype=dtype).reshape(len(rows), top_k)
    if perm is None:
        indices = torch.arange(n, dtype=torch.int32)
    else:
        indices = torch.tensor(perm, dtype=torch.int32)
    scratch = lambda size: torch.full((size,), -123, dtype=torch.int32)
    return {
        "topk_ids": topk_ids,
        "token_expert_indices": indices.reshape(len(rows), top_k),
        "expert_map": None if mapping is None else torch.tensor(mapping, dtype=torch.int32),
        "n_expert": n_expert,
        "n_local_expert": n_local,
        "expert_first_token_offset": torch.full((n_local + 1,), -91, dtype=torch.int64),
        "scatter_idx": scratch(n),
        "sort_workspace": torch.full((1 << 18,), 71, dtype=torch.int8),
        "sorted_experts": scratch(n + 7),
        "sorted_rows": scratch(n + 7),
        "topk_ids_for_sort": scratch(n + 7),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_moe_prepare_scatter(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_moe_prepare_scatter(**inputs)


def _check_offsets(offsets, size):
    assert offsets.dtype == torch.int64
    assert int(offsets[0]) == 0
    values = offsets.tolist()
    assert values == sorted(values), "expert offsets must be monotone"
    assert 0 <= values[-1] <= size


def compute_ref_performance(inputs):
    """Time the reference on independently rebuilt inputs."""
    return measure(compute_ref, lambda: fresh(inputs, kernelgenbench.device))


def compute_res_performance(inputs):
    """Time the candidate on independently rebuilt inputs."""
    return measure(compute_res, lambda: fresh(inputs, kernelgenbench.device))


@label("v1_vllm_moe_prepare_scatter")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_moe_prepare_scatter(config):
    rows, top_k, _, n_local, _, dtype, _ = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before_ids = res_inputs["topk_ids"].clone()
    before_indices = res_inputs["token_expert_indices"].clone()
    size = res_inputs["topk_ids"].numel()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)

    # Public return structure and identity.
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    assert res_out[0] is res_inputs["expert_first_token_offset"]
    assert torch._C._is_alias_of(res_out[1], res_inputs["scatter_idx"])
    assert res_out[1].shape == res_inputs["topk_ids"].shape
    assert torch.equal(ref_out[0], res_out[0])
    assert torch.equal(ref_out[1].reshape(-1), res_out[1].reshape(-1))
    _check_offsets(res_out[0], size)

    # Scratch prefixes are contractual where the caller observes them; tails
    # beyond numel() must stay untouched.
    for key in ("sorted_experts", "sorted_rows", "topk_ids_for_sort"):
        assert torch.equal(ref_inputs[key][:size], res_inputs[key][:size]), key
        assert bool((res_inputs[key][size:] == -123).all()), key
    if size:
        inverse = res_out[1].reshape(-1)
        assert sorted(inverse.tolist()) == list(range(size)), "inverse must be a permutation"
    assert torch.equal(before_ids, res_inputs["topk_ids"]), "topk_ids must not change"
    assert torch.equal(before_indices, res_inputs["token_expert_indices"]), "indices must not change"

    ref_ms = compute_ref_performance(inputs)
    res_ms = compute_res_performance(inputs)
    return result(ref_ms, res_ms)
