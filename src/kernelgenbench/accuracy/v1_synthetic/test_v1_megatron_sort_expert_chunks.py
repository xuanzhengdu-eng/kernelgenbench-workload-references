"""Accuracy and performance for v1_megatron_sort_expert_chunks.

Contract: ``sort_chunks_by_idxs`` eager path. ``input`` rows are split into
chunks of ``split_sizes`` and concatenated in ``sorted_idxs`` order; the optional
``probs`` tensor is split and reordered identically. Empty chunks are legal and
explicitly covered; inputs are not modified and outputs never alias them.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import moe_rows222_231_checks as checks


# (chunk lengths, chunk permutation, width, dtype, probabilities present)
COMBOS = [
    ((2, 0, 3), (2, 0, 1), 8, torch.float32, True),
    ((1, 3, 2), (1, 2, 0), 7, torch.bfloat16, False),
    ((0, 0, 0), (2, 1, 0), 4, torch.float32, True),
    ((5,), (0,), 16, torch.bfloat16, True),
    ((1, 1, 1, 1), (3, 0, 2, 1), 5, torch.float16, True),
    ((0,), (0,), 3, torch.float32, False),
    ((2, 4), (0, 1), 1, torch.float32, True),
]


def input_build(config):
    sizes, order, width, dtype, has_probs = config
    generator = torch.Generator(device="cpu").manual_seed(225)
    total = sum(sizes)
    return {
        "input": torch.randn((total, width), dtype=dtype, generator=generator),
        "split_sizes": torch.tensor(sizes, dtype=torch.int64),
        "sorted_idxs": torch.tensor(order, dtype=torch.int64),
        "probs": torch.rand((total, 1), generator=generator) if has_probs else None,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_sort_expert_chunks(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_sort_expert_chunks(**inputs)


def compare(ref_out, res_out, dtype=None, has_probs=False):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 2
    assert ref_out[0].shape == res_out[0].shape and res_out[0].dtype == dtype
    torch.testing.assert_close(res_out[0], ref_out[0], rtol=0, atol=0)
    if not has_probs:
        assert ref_out[1] is None and res_out[1] is None
    else:
        assert ref_out[1].shape == res_out[1].shape
        torch.testing.assert_close(res_out[1], ref_out[1], rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_megatron_sort_expert_chunks, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_megatron_sort_expert_chunks, inputs, device=kernelgenbench.device)


@label("v1_megatron_sort_expert_chunks")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_sort_expert_chunks(config):
    sizes, order, width, dtype, has_probs = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    before = {
        name: value.cpu().clone()
        for name, value in res_inputs.items()
        if isinstance(value, torch.Tensor)
    }

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype, has_probs)

    # Direct contract assertion: concatenate the explicitly selected chunks.
    source = res_inputs["input"]
    chunks = torch.split(source, sizes, dim=0)
    expected = torch.cat([chunks[index] for index in order], dim=0)
    assert torch.equal(res_out[0], expected)
    if has_probs:
        prob_chunks = torch.split(res_inputs["probs"], sizes, dim=0)
        assert torch.equal(res_out[1], torch.cat([prob_chunks[index] for index in order], dim=0))
    outputs = (res_out[0],) if not has_probs else (res_out[0], res_out[1])
    for output in outputs:
        if output.numel():
            assert output.data_ptr() not in {tensor.data_ptr() for tensor in res_inputs.values() if isinstance(tensor, torch.Tensor)}
    for name, value in before.items():
        assert torch.equal(res_inputs[name].cpu(), value), f"{name} was modified"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
