"""Accuracy and performance for v1_tensor_masked_update (Megatron Triton op).

Contract: states=(N,*tail) 2-D/3-D/4-D (possibly strided), idx=(B,) int32,
new_states=(B,*tail).  For each i, new_states[i] overwrites states[idx[i]]
unless idx[i] == -1, which is skipped.  Returns None, mutates only states, and
preserves the caller's strides.
"""
import statistics

import kernelgenbench
import pytest
import torch

from kernelgenbench.accuracy.v1_synthetic import bias_activation_test_utils as checks
from sandbox.verifier.test_parametrize import label, parametrize


CASES = [(11,), (3, 7), (2, 3, 5), (3, 513)]
COMBOS = [
    (case, dtype, strided)
    for case in CASES
    for dtype in (torch.float32, torch.bfloat16)
    for strided in (False, True)
]
INDEX = [4, -1, 1, 0]


def input_build(config):
    case, dtype, strided = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)

    def make(batch):
        width = case[-1] * (2 if strided else 1)
        tensor = torch.randn((batch, *case[:-1], width), dtype=dtype,
                             generator=generator, device="cpu")
        return tensor[..., ::2] if strided else tensor

    return {
        "states": make(6),
        "idx": torch.tensor(INDEX, dtype=torch.int32),
        "new_states": make(4),
    }


def compute_ref(states, idx, new_states):
    return kernelgenbench.baseline.v1_tensor_masked_update(states, idx, new_states)


def compute_res(states, idx, new_states):
    return kernelgenbench.solution.v1_tensor_masked_update(states, idx, new_states)


def compare(ref_out, res_out, dtype=None):
    assert ref_out is None and res_out is None, "tensor_masked_update returns None"


def expected_states(states, idx, new_states):
    expected = states.clone()
    for i in range(idx.numel()):
        destination = int(idx[i])
        if destination != -1:
            expected[destination] = new_states[i]
    return expected


def fresh(inputs, device):
    """Device copy that preserves the caller's strided layout."""
    copied = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            target = torch.empty_strided(value.shape, value.stride(),
                                         dtype=value.dtype, device=device)
            target.copy_(value)
            copied[name] = target
        else:
            copied[name] = value
    return copied


def measure(fn, inputs):
    """Median single-call milliseconds with layout-preserving input reset."""
    samples = []
    for _ in range(6):
        arguments = fresh(inputs, kernelgenbench.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(**arguments)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples[1:])


def compute_ref_performance(states, idx, new_states):
    return measure(compute_ref, {"states": states, "idx": idx,
                                 "new_states": new_states})


def compute_res_performance(states, idx, new_states):
    return measure(compute_res, {"states": states, "idx": idx,
                                 "new_states": new_states})


@label("v1_tensor_masked_update")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_tensor_masked_update(config):
    case, dtype, strided = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    # The caller's strided layout must survive the device copy and the update.
    expected_stride = inputs["states"].stride()
    assert ref_inputs["states"].stride() == expected_stride
    assert res_inputs["states"].stride() == expected_stride

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    expected = expected_states(inputs["states"], inputs["idx"], inputs["new_states"])
    torch.testing.assert_close(res_inputs["states"].cpu(), expected, rtol=0, atol=0)
    assert res_inputs["states"].stride() == expected_stride, "update changed strides"
    # idx == -1 is a skip: row N-1 must stay untouched.
    untouched_row = inputs["states"][inputs["states"].shape[0] - 1]
    torch.testing.assert_close(res_inputs["states"][-1].cpu(), untouched_row, rtol=0, atol=0)
    # Other tensors are read-only.
    torch.testing.assert_close(res_inputs["idx"].cpu(), inputs["idx"], rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["new_states"].cpu(), inputs["new_states"],
                               rtol=0, atol=0)

    return checks.timings(
        compute_ref_performance(**inputs),
        compute_res_performance(**inputs),
    )
