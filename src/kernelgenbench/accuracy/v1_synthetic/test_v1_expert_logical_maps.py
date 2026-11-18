"""Accuracy and performance for v1_expert_logical_maps.

Official entry: ``vllm.distributed.eplb.eplb_state.compute_logical_maps``
(vLLM f358f0578a83f4e1fe6fc49a492d3ae9362de8e5, eplb_state.py L1316-L1390).
The input is CPU ``(P,)`` or ``(L, P)`` integer data; negative ids are inactive.
The inverse map keeps ascending physical-slot order per logical expert, pads
unused replicas with -1, and returns the replica count alongside.  int32 and
int64 both work and the input is never modified.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.moe_rows192_201_utils import (
    fresh,
    measure,
    timings,
)
from sandbox.verifier.test_parametrize import label, parametrize


# (physical_to_logical values, num logical experts, dtype)
COMBOS = [
    ([1, 0, 1, -1, 1, 0], 3, torch.int64),
    ([[2, 0, 2, -1], [1, 1, -3, 0]], 3, torch.int64),
    ([], 3, torch.int64),
    ([-1, -1, -1], 3, torch.int64),
    ([0, 1, 0], 3, torch.int32),
    ([[0, 1, 2], [2, 1, 0]], 3, torch.int32),
    ([0, 0, 0, 1], 2, torch.int32),
]


def input_build(config):
    values, num_logical_experts, dtype = config
    return {
        "physical_to_logical_map": torch.tensor(values, dtype=dtype, device="cpu"),
        "num_logical_experts": num_logical_experts,
    }


def compute_ref(physical_to_logical_map, num_logical_experts):
    return kernelgenbench.baseline.v1_expert_logical_maps(
        physical_to_logical_map, num_logical_experts
    )


def compute_res(physical_to_logical_map, num_logical_experts):
    return kernelgenbench.solution.v1_expert_logical_maps(
        physical_to_logical_map, num_logical_experts
    )


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 2
    for expected, actual in zip(ref_out, res_out):
        assert isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor)
        assert expected.shape == actual.shape
        assert expected.dtype == actual.dtype == dtype
        assert expected.device == actual.device
        assert torch.equal(expected, actual)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs, device=torch.device("cpu"))


def compute_res_performance(**inputs):
    return measure(compute_res, inputs, device=torch.device("cpu"))


@label("v1_expert_logical_maps")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_expert_logical_maps(config):
    values, num_logical_experts, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs)
    res_inputs = fresh(inputs)
    before = res_inputs["physical_to_logical_map"].clone()
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    assert torch.equal(res_inputs["physical_to_logical_map"], before)
    expected_replicas = int(ref_out[1].max().item()) if ref_out[1].numel() else 0
    assert ref_out[0].shape[-1] == expected_replicas

    # The inverse keeps ascending physical slots per logical expert and pads the
    # unused replicas with -1; check both sides for every layer and expert.
    per_layer = inputs["physical_to_logical_map"].dim() == 1
    for inverse, counts in (ref_out, res_out):
        layers = inverse.unsqueeze(0) if per_layer else inverse
        layer_counts = counts.unsqueeze(0) if per_layer else counts
        for layer in range(layers.shape[0]):
            for expert in range(layers.shape[1]):
                used = int(layer_counts[layer, expert])
                slots = layers[layer, expert, :used]
                assert torch.equal(slots, torch.sort(slots).values), "physical slots must ascend"
                assert bool((layers[layer, expert, used:] == -1).all()), "unused replicas must be -1"
                if used:
                    assert int(slots.max()) < inputs["physical_to_logical_map"].numel()

    # Invalid shapes and ids fail in the source; the reference mirrors the error class.
    if config is COMBOS[0]:
        bad_cases = [
            (torch.tensor(1, dtype=torch.int64), 3),
            (torch.zeros((2, 2, 2), dtype=torch.int64), 3),
            (torch.tensor([0, 3], dtype=torch.int64), 3),
            (torch.tensor([0, 1], dtype=torch.int64), 0),
        ]
        for bad, count in bad_cases:
            for fn in (compute_ref, compute_res):
                try:
                    fn(physical_to_logical_map=bad, num_logical_experts=count)
                except (AssertionError, RuntimeError):
                    pass
                else:
                    raise AssertionError(f"invalid map accepted: {bad!r}, {count}")

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
