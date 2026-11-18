"""Accuracy and performance for the Megatron unfused CE explicit backward.

``softmax`` is the saved FP32 local softmax from the forward pass.  The
official backward rewrites it in place and returns that same buffer as the
gradient, scaled by ``grad_output``; rows whose target lies outside the shard
(``target_mask``) keep the softmax (no one-hot subtraction).  Label smoothing
subtracts ``smoothing / V`` from every entry and scales the target entry by
``1 - smoothing``.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (rows, local vocab, label smoothing)
COMBOS = [((4, 8), 0.0), ((3, 16), 0.0), ((4, 8), 0.1)]


def input_build(config):
    (rows, n), smoothing = config
    generator = torch.Generator(device="cpu").manual_seed(20261113)
    logits = torch.randn((rows, n), generator=generator).float()
    target = torch.randint(0, n, (rows,), generator=generator)
    target[1] = -100
    # Reproduce the forward's saved softmax exactly (max-shift, exp, normalize).
    x = logits.clone()
    x -= x.max(dim=-1).values.unsqueeze(-1)
    x.exp_()
    x /= x.sum(dim=-1, keepdim=True)
    mask = (target < 0) | (target >= n)
    masked_target = target.clone()
    masked_target[mask] = 0
    return {
        "softmax": x,
        "target_mask": mask,
        "masked_target": masked_target,
        "grad_output": torch.linspace(0.5, 1.5, rows),
        "label_smoothing": smoothing,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_vocab_parallel_ce_backward_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_vocab_parallel_ce_backward_(**inputs)


def compare(ref_out, res_out, shape):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert ref_out.shape == res_out.shape == shape
    assert ref_out.dtype == res_out.dtype == torch.float32
    torch.testing.assert_close(res_out, ref_out, rtol=1e-5, atol=1e-6)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_megatron_vocab_parallel_ce_backward_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_vocab_parallel_ce_backward_(config):
    (rows, n), smoothing = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    saved = inputs["softmax"].to(kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, (rows, n))

    # Both sides return the saved-softmax buffer itself, rewritten in place.
    assert ref_out.data_ptr() == ref_inputs["softmax"].data_ptr()
    assert res_out.data_ptr() == res_inputs["softmax"].data_ptr()
    assert not torch.equal(ref_inputs["softmax"], saved)
    for side in (ref_inputs, res_inputs):
        for key in ("target_mask", "masked_target"):
            assert torch.equal(side[key], inputs[key].to(kernelgenbench.device)), key

    # Independent oracle for the unsmoothed case.
    if smoothing == 0.0:
        target = inputs["masked_target"].to(kernelgenbench.device)
        expected = saved - torch.nn.functional.one_hot(target, n).float()
        in_shard = (~inputs["target_mask"]).to(kernelgenbench.device)
        expected = torch.where(in_shard.unsqueeze(-1), expected, saved)
        expected = expected * inputs["grad_output"].to(kernelgenbench.device).unsqueeze(-1)
        torch.testing.assert_close(ref_out, expected, rtol=1e-5, atol=1e-6)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
