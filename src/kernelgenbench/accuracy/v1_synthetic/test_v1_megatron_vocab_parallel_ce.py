"""Accuracy and performance for the Megatron unfused vocab-parallel CE forward.

``logits`` is one local ``[rows, local_vocab]`` shard, ``target`` holds global
ids and ids outside ``[0, local_vocab)`` (here -100) are masked.  The FP32
working copy is rewritten in place to the local softmax, so an FP32 input is
observably mutated while FP16/BF16 inputs are not.  The loss is FP32 and keeps
the leading shape of ``target``.  ``label_smoothing`` follows the upstream
``(1 - s) * loss - s * mean(log_softmax)`` formula.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (rows, local vocab, dtype, label smoothing)
COMBOS = [
    ((2, 4, 8), torch.float32, 0.0),
    ((3, 8), torch.float16, 0.0),
    ((3, 8), torch.bfloat16, 0.0),
    ((4, 8), torch.float32, 0.1),
    ((2, 16), torch.float32, 0.5),
]


def input_build(config):
    shape, dtype, smoothing = config
    generator = torch.Generator(device="cpu").manual_seed(20261111)
    logits = torch.randn(shape, dtype=dtype, generator=generator)
    target = torch.randint(0, shape[-1], shape[:-1], generator=generator)
    target.reshape(-1)[1] = -100   # out-of-shard id: no target logit
    return {"logits": logits, "target": target, "label_smoothing": smoothing}


def compute_ref(logits, target, label_smoothing):
    return kernelgenbench.baseline.v1_megatron_vocab_parallel_ce(
        logits, target, label_smoothing)


def compute_res(logits, target, label_smoothing):
    return kernelgenbench.solution.v1_megatron_vocab_parallel_ce(
        logits, target, label_smoothing)


def compare(ref_out, res_out, dtype, target):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert ref_out.shape == res_out.shape == target.shape
    assert ref_out.dtype == res_out.dtype == torch.float32
    rtol, atol = {
        torch.float32: (1e-5, 1e-6),
        torch.float16: (2e-3, 1e-3),
        torch.bfloat16: (1e-2, 8e-3),
    }[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(logits, target, label_smoothing):
    return measure(compute_ref, {"logits": logits, "target": target,
                                 "label_smoothing": label_smoothing})


def compute_res_performance(logits, target, label_smoothing):
    return measure(compute_res, {"logits": logits, "target": target,
                                 "label_smoothing": label_smoothing})


@label("v1_megatron_vocab_parallel_ce")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_vocab_parallel_ce(config):
    shape, dtype, smoothing = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before = ref_inputs["logits"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype, ref_inputs["target"])

    # The FP32 working copy is the caller's tensor: it becomes the softmax.
    for side in (ref_inputs, res_inputs):
        if dtype == torch.float32:
            assert not torch.equal(side["logits"], before)
            assert torch.allclose(side["logits"].float().sum(-1),
                                  torch.ones(shape[:-1], device=side["logits"].device),
                                  rtol=1e-5, atol=1e-5)
        else:
            assert torch.equal(side["logits"], before)
    torch.testing.assert_close(ref_inputs["logits"], res_inputs["logits"],
                               rtol=1e-6, atol=1e-7)

    # Independent oracle: a single shard is an ordinary cross entropy.
    if smoothing == 0.0:
        n = shape[-1]
        joined = before.float().view(-1, n)
        target = inputs["target"].to(kernelgenbench.device).view(-1)
        oracle = torch.nn.functional.cross_entropy(
            joined, target, reduction="none", ignore_index=-100)
        # Megatron still returns logZ for masked rows; PyTorch returns 0 there,
        # so the independent oracle only covers rows with an in-shard target.
        covered = target >= 0
        torch.testing.assert_close(ref_out.view(-1)[covered], oracle[covered],
                                   rtol=1e-5, atol=1e-6)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
