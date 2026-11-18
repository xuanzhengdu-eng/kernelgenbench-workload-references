"""Accuracy and performance for the Megatron fused CE explicit backward.

The fused backward performs the same in-place FP32 rewrite of the saved
softmax as the unfused path, but returns a *new* BF16 tensor (the JIT-fused
helper casts its result).  No label smoothing upstream.  The returned dtype and
the aliasing therefore differ from the unfused entry, which is why it is a
separate Item.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (rows, local vocab)
COMBOS = [((4, 8),), ((3, 16),), ((2, 5),)]


def input_build(config):
    (rows, n), = config
    generator = torch.Generator(device="cpu").manual_seed(20261114)
    logits = torch.randn((rows, n), generator=generator).float()
    target = torch.randint(0, n, (rows,), generator=generator)
    target[0] = n + 3          # out-of-shard id -> masked row
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
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_vocab_parallel_ce_backward_fused(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_vocab_parallel_ce_backward_fused(**inputs)


def compare(ref_out, res_out, shape):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert ref_out.shape == res_out.shape == shape
    assert ref_out.dtype == res_out.dtype == torch.bfloat16
    torch.testing.assert_close(res_out, ref_out, rtol=1.6e-2, atol=8e-3)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_megatron_vocab_parallel_ce_backward_fused")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_vocab_parallel_ce_backward_fused(config):
    (rows, n), = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    saved = inputs["softmax"].to(kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, (rows, n))

    # A new BF16 tensor is returned rather than the saved buffer itself.
    assert ref_out.data_ptr() != ref_inputs["softmax"].data_ptr()
    assert res_out.data_ptr() != res_inputs["softmax"].data_ptr()
    assert not torch.equal(ref_inputs["softmax"], saved)
    assert not torch.equal(res_inputs["softmax"], saved)
    # The candidate path carries the gradient through the official BF16 cast, so
    # the written-back buffer matches to BF16 precision.
    torch.testing.assert_close(ref_inputs["softmax"], res_inputs["softmax"],
                               rtol=1.6e-2, atol=8e-3)
    for side in (ref_inputs, res_inputs):
        for key in ("target_mask", "masked_target"):
            assert torch.equal(side[key], inputs[key].to(kernelgenbench.device)), key

    # The mutated FP32 buffer is the unfused analytic gradient.
    target = inputs["masked_target"].to(kernelgenbench.device)
    expected = saved - torch.nn.functional.one_hot(target, n).float()
    in_shard = (~inputs["target_mask"]).to(kernelgenbench.device)
    expected = torch.where(in_shard.unsqueeze(-1), expected, saved)
    expected = (expected * inputs["grad_output"].to(kernelgenbench.device).unsqueeze(-1))
    torch.testing.assert_close(ref_inputs["softmax"], expected, rtol=1e-5, atol=1e-6)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
