"""Accuracy and performance for the DeepSpeed saved-softmax CE backward.

The reference reproduces DeepSpeed's explicit backward contract: the input is
the local (globally normalized) softmax shard, ``target_mask`` marks rows whose
target is outside this shard or ignored, ``masked_target`` is the in-shard
target index, ``valid_target`` marks rows that contribute at all, and the
result is a *new* tensor scaled by ``grad_output`` and cast to
``logits_dtype``.  Out-of-shard but valid rows keep the softmax (no one-hot
subtraction); ignored rows are zero.

The candidate is the official DeepSpeed autograd backward, reached through the
public ``vocab_parallel_cross_entropy`` entry: the test's softmax is a
normalized distribution, so ``log(softmax)`` reproduces the upstream saved
softmax and ``loss.backward(grad_output)`` runs the upstream backward kernel.
Out-of-shard rows are produced by placing the shard at a non-zero vocabulary
offset, which selects the official ``target_in_partition=False`` branch.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (world size, rank, rows, local vocab, label smoothing, result dtype)
COMBOS = [
    (1, 0, 4, 8, 0.0, torch.float32),
    (2, 0, 3, 8, 0.0, torch.float32),
    (2, 1, 4, 8, 0.0, torch.bfloat16),
]


def input_build(config):
    world, rank, rows, n, smoothing, dtype = config
    generators = [torch.Generator(device="cpu").manual_seed(20261110 + r)
                  for r in range(world)]
    all_logits = torch.cat([
        torch.randn((rows, n), generator=g).to(dtype).float() for g in generators
    ], dim=-1)
    local_logits = all_logits[:, rank * n:(rank + 1) * n]
    # A normalized local distribution, as saved by the upstream forward.
    softmax = local_logits.softmax(dim=-1).contiguous()
    target = torch.tensor([i * (world * n - 1) // max(rows - 1, 1) for i in range(rows)])
    target[1] = -100
    valid_target = target != -100
    target_mask = (target < rank * n) | (target >= (rank + 1) * n) | (~valid_target)
    masked_target = target - rank * n
    masked_target[target_mask] = 0
    return {
        "softmax": softmax,
        "target_mask": target_mask,
        "masked_target": masked_target.reshape(-1),
        "grad_output": torch.linspace(0.5, 1.5, rows),
        "label_smoothing": smoothing,
        "backend": "deepspeed",
        "valid_target": valid_target,
        "logits_dtype": dtype,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_vocab_parallel_ce_backward_deepspeed(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_vocab_parallel_ce_backward_deepspeed(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert ref_out.shape == res_out.shape
    assert ref_out.dtype == res_out.dtype == dtype
    rtol, atol = {
        torch.float32: (1e-5, 1e-5),
        torch.bfloat16: (1.6e-2, 8e-3),
    }[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_megatron_vocab_parallel_ce_backward_deepspeed")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_vocab_parallel_ce_backward_deepspeed(config):
    _world, _rank, rows, n, _smoothing, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["softmax"].shape == (rows, n)
        assert side["softmax"].dtype == torch.float32
        assert side["masked_target"].shape == (rows,)
        assert side["target_mask"].dtype == torch.bool

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # The DeepSpeed path returns a fresh tensor and never mutates the saved
    # softmax; masks and upstream gradients are read-only.
    assert ref_out.data_ptr() != ref_inputs["softmax"].data_ptr()
    assert res_out.data_ptr() != res_inputs["softmax"].data_ptr()
    expected_softmax = inputs["softmax"].to(kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["softmax"], expected_softmax, rtol=0, atol=0)
        for key in ("target_mask", "masked_target"):
            assert torch.equal(side[key], inputs[key].to(kernelgenbench.device)), key
        torch.testing.assert_close(side["grad_output"],
                                   inputs["grad_output"].to(kernelgenbench.device),
                                   rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
