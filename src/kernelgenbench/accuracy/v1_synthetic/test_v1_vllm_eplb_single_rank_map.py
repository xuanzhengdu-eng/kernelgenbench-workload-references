"""Accuracy and performance for v1_vllm_eplb_single_rank_map.

Contract: the single-rank specialization of
``DefaultEplbPolicy.rebalance_experts`` (one group per logical expert, one node,
one rank). Replicas are assigned by largest ``load / replica_count`` and the
physical slots are ordered by descending effective load; an optional old map
preserves slots that stay on the rank. Everything is CPU arithmetic and returns
an int64 CPU tensor; no weights are moved.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import moe_rows222_231_checks as checks


# (per-layer loads, total physical replicas, old mapping or None)
COMBOS = [
    ([[8.0, 3.0, 1.0]], 3, None),
    ([[9.0, 3.0, 1.0]], 6, None),
    ([[4.0, 4.0, 2.0, 2.0]], 8, None),
    ([[9.0, 3.0, 1.0]], 6, [[2, 0, 1, 0, 0, 1]]),
    ([[0.0, 0.0, 0.0]], 5, [[2, 1, 0, 1, 2]]),
    ([[10.0, 3.0, 1.0], [1.0, 3.0, 9.0]], 7, [[1, 0, 0, 0, 2, 2, 1], [0, 1, 2, 2, 2, 2, 1]]),
    ([[1.0], [5.0]], 4, [[0, 0, 0, 0], [0, 0, 0, 0]]),
    ([[0.0, 0.0, 4.0, 0.0]], 4, None),
]


def input_build(config):
    loads, replicas, old = config
    return {
        "weight": torch.tensor(loads, dtype=torch.float32),
        "num_replicas": replicas,
        "old_global_expert_indices": None
        if old is None
        else torch.tensor(old, dtype=torch.int64),
    }


def compute_ref(weight, num_replicas, old_global_expert_indices=None):
    return kernelgenbench.baseline.v1_vllm_eplb_single_rank_map(
        weight, num_replicas, old_global_expert_indices
    )


def compute_res(weight, num_replicas, old_global_expert_indices=None):
    return kernelgenbench.solution.v1_vllm_eplb_single_rank_map(
        weight, num_replicas, old_global_expert_indices
    )


def compare(ref_out, res_out, dtype=None):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == torch.int64
    assert res_out.device.type == "cpu"
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def compute_ref_performance(weight, num_replicas, old_global_expert_indices=None):
    return checks.measure(
        kernelgenbench.baseline.v1_vllm_eplb_single_rank_map,
        {
            "weight": weight,
            "num_replicas": num_replicas,
            "old_global_expert_indices": old_global_expert_indices,
        },
    )


def compute_res_performance(weight, num_replicas, old_global_expert_indices=None):
    return checks.measure(
        kernelgenbench.solution.v1_vllm_eplb_single_rank_map,
        {
            "weight": weight,
            "num_replicas": num_replicas,
            "old_global_expert_indices": old_global_expert_indices,
        },
    )


@label("v1_vllm_eplb_single_rank_map")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_eplb_single_rank_map(config):
    loads, replicas, old = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, "cpu")
    res_inputs = checks.fresh(inputs, "cpu")
    before = {
        name: value.clone()
        for name, value in res_inputs.items()
        if isinstance(value, torch.Tensor)
    }

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    experts = len(loads[0])
    # Contract: each logical expert owns at least one slot, every slot is a
    # valid logical id, and each row has exactly num_replicas entries.
    assert res_out.shape == (len(loads), replicas)
    assert res_out.min().item() >= 0 and res_out.max().item() < experts
    for layer in range(len(loads)):
        counts = torch.bincount(res_out[layer], minlength=experts)
        assert bool((counts >= 1).all()), "every logical expert must keep a slot"
    for name, value in before.items():
        assert torch.equal(res_inputs[name], value), f"{name} was modified"

    # CPU policy orchestration; no GPU kernel timing is claimed.
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
