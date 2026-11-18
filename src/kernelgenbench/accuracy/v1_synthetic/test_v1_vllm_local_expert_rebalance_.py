"""Accuracy and performance for v1_vllm_local_expert_rebalance_ (vLLM EPLB).

Official entry: vLLM ``rearrange_expert_weights_inplace``, pinned commit
f358f0578a83f4e1fe6fc49a492d3ae9362de8e5, lines 669-769, with the one-rank
local staging path in ``move_to_buffer``/``move_from_buffer``.  The candidate
binding supplies a one-rank process-group facade and a no-op communicator, so
the official Python entry performs the same local copies.

Input: one or two layers of ``old_indices``/``new_indices``
[num_layers, num_physical] global->logical maps, two weight tensors per layer
[E, 3, 4] and [E, 5], and one staging buffer per weight.  Each destination
slot receives the weight of logical expert ``new[dst]``; -1 slots are skipped
and duplicate destinations are allowed.  ``is_profile=True`` performs no copy.
This test covers the one-rank path; the two-rank communicator path is not
claimed.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows202_211_test_utils import fresh, measure, result


# old maps, new maps, dtype, profile
COMBOS = [
    ([[0, 1, 2, 3]], [[2, 0, 3, 1]], torch.float32, False),
    ([[0, 1, 2, 3]], [[3, 2, 1, 0]], torch.float32, False),
    ([[0, 1, 2, 3]], [[0, 1, 2, -1]], torch.float32, False),
    ([[0, 1, 2, 3]], [[0, 0, 2, 3]], torch.float32, False),
    ([[0, 1, 2, 3], [4, 5, 6, 7]], [[3, 2, 1, 0], [6, 5, 4, 7]], torch.bfloat16, False),
    ([[0, 1, 2, 3]], [[2, 0, 3, 1]], torch.bfloat16, True),
]


def input_build(config):
    old_layers, new_layers, dtype, profile = config
    weights = []
    for layer in old_layers:
        tensors = []
        for shape in ((3, 4), (5,)):
            rows = []
            for expert in layer:
                generator = torch.Generator(device="cpu").manual_seed(
                    20261110 + max(expert, 0) * 13 + len(shape)
                )
                rows.append(torch.randn(shape, dtype=dtype, generator=generator))
            tensors.append(torch.stack(rows))
        weights.append(tensors)
    buffers = [torch.full_like(weights[0][0], -17), torch.full_like(weights[0][1], -19)]
    return {
        "old_indices": torch.tensor(old_layers, dtype=torch.int64),
        "new_indices": torch.tensor(new_layers, dtype=torch.int64),
        "expert_weights": weights,
        "expert_buffer": buffers,
        "is_profile": profile,
        "ep_size": 1,
        "ep_rank": 0,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_local_expert_rebalance_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_local_expert_rebalance_(**inputs)


def _expected_weights(old_layers, new_layers, before):
    expected = []
    for layer, (old, new) in enumerate(zip(old_layers, new_layers)):
        # Build each weight tensor independently from the same source rows.
        layer_expected = []
        for weight_index in range(len(before[layer])):
            rows = []
            for dst, expert in enumerate(new):
                source = dst if expert == -1 else old.index(expert)
                rows.append(before[layer][weight_index][source].clone())
            layer_expected.append(torch.stack(rows))
        expected.append(layer_expected)
    return expected


def compute_ref_performance(inputs):
    """Time the reference on independently rebuilt inputs."""
    return measure(compute_ref, lambda: fresh(inputs, kernelgenbench.device))


def compute_res_performance(inputs):
    """Time the candidate on independently rebuilt inputs."""
    return measure(compute_res, lambda: fresh(inputs, kernelgenbench.device))


@label("v1_vllm_local_expert_rebalance_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_local_expert_rebalance_(config):
    old_layers, new_layers, dtype, profile = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before = [[w.clone() for w in layer] for layer in res_inputs["expert_weights"]]
    before_buffers = [b.clone() for b in res_inputs["expert_buffer"]]
    before_old = res_inputs["old_indices"].clone()
    before_new = res_inputs["new_indices"].clone()

    assert compute_ref(**ref_inputs) is None
    assert compute_res(**res_inputs) is None

    for ref_layer, res_layer in zip(ref_inputs["expert_weights"], res_inputs["expert_weights"]):
        for ref_weight, res_weight in zip(ref_layer, res_layer):
            assert res_weight.shape == ref_weight.shape
            assert res_weight.dtype == ref_weight.dtype
            assert torch.equal(res_weight, ref_weight), "reference and solution weights differ"
    for ref_buf, res_buf in zip(ref_inputs["expert_buffer"], res_inputs["expert_buffer"]):
        assert torch.equal(res_buf, ref_buf), "staging buffer differs"

    if profile:
        for layer, saved in zip(res_inputs["expert_weights"], before):
            for weight, original in zip(layer, saved):
                assert torch.equal(weight, original), "profile must not copy weights"
        for buffer, original in zip(res_inputs["expert_buffer"], before_buffers):
            assert torch.equal(buffer, original), "profile must not touch buffers"
    else:
        expected = _expected_weights(old_layers, new_layers, before)
        for layer, expected_layer in zip(res_inputs["expert_weights"], expected):
            for weight, expected_weight in zip(layer, expected_layer):
                assert torch.equal(weight, expected_weight), "destination row has wrong source expert"
    assert torch.equal(res_inputs["old_indices"], before_old)
    assert torch.equal(res_inputs["new_indices"], before_new)

    ref_ms = compute_ref_performance(inputs)
    res_ms = compute_res_performance(inputs)
    return result(ref_ms, res_ms)
