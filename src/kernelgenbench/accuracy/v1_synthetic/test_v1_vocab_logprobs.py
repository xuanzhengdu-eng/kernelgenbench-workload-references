"""Accuracy and performance for v1_vocab_logprobs.

DeepSpeed ``_VocabParallelLogProbs.forward`` over one vocabulary shard:
``logits`` is ``(batch, time, vocab)`` or ``(rows, vocab)`` in fp32/fp16/bf16,
``index`` is ``(rows, 3)`` int64 containing repeated ids and ``ignore_index``,
and the output is ``(rows, 3)`` fp32.  Ignored ids read 0.0; an all ``-inf`` row
produces ``-inf`` for the requested ids (its softmax is undefined) and logits and
index are left untouched.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows172_181_test_utils import measure_cuda
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (shape, dtype)
COMBOS = [
    ((2, 4, 32), torch.float32),
    ((5, 67), torch.float32),
    ((2, 4, 32), torch.float16),
    ((5, 67), torch.float16),
    ((3, 3, 16), torch.bfloat16),
    ((1, 129), torch.bfloat16),
]


def input_build(config):
    shape, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    logits = torch.randn(shape, dtype=dtype, generator=generator)
    rows = logits.reshape(-1, shape[-1])
    # Row 0 is undefined (all -inf); row 2 masks one vocabulary entry.
    rows[0, :] = float("-inf")
    if rows.shape[0] > 2:
        rows[2, 0] = float("-inf")
    index = torch.randint(0, shape[-1], (rows.shape[0], 3), generator=generator)
    index[:, 1] = index[:, 0]
    if rows.shape[0] > 1:
        index[1, :] = -100
    if rows.shape[0] > 2:
        index[2, 2] = -100
    return {"logits": logits, "index": index, "ignore_index": -100}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vocab_logprobs(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vocab_logprobs(**inputs)


def compare(ref_out, res_out, dtype):
    assert ref_out.shape == res_out.shape and ref_out.dtype == res_out.dtype == torch.float32
    torch.testing.assert_close(res_out, ref_out, atol=1e-5, rtol=1e-5, equal_nan=True)


def compute_ref_performance(inputs):
    """Time the reference entry on the prepared inputs (read-only operator)."""
    return measure_cuda(lambda: compute_ref(**inputs))


def compute_res_performance(inputs):
    """Time the solution entry on the same prepared inputs."""
    return measure_cuda(lambda: compute_res(**inputs))


@label("v1_vocab_logprobs")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vocab_logprobs(config):
    shape, dtype = config
    inputs = input_build(config)
    saved = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    cuda = {name: (value.cuda() if isinstance(value, torch.Tensor) else value)
            for name, value in inputs.items()}

    ref_out = compute_ref(**cuda)
    res_out = compute_res(**cuda)
    compare(ref_out, res_out, dtype)

    rows, vocab = cuda["logits"].reshape(-1, shape[-1]).shape
    valid = cuda["index"] != cuda["ignore_index"]
    expected = torch.log_softmax(cuda["logits"].reshape(rows, vocab).float(), dim=-1).gather(
        1, cuda["index"].clamp(0, vocab - 1))
    expected = torch.where(valid, expected, torch.zeros_like(expected))
    torch.testing.assert_close(res_out, expected, atol=1e-6, rtol=1e-6, equal_nan=True)
    # Ignored ids are exactly zero, not a gathered -inf.
    assert torch.equal(res_out[~valid], torch.zeros_like(res_out[~valid]))
    # A row whose logits are all -inf has no softmax: the official forward yields NaN there.
    assert torch.isnan(res_out[0]).all(), res_out[0]
    for name, value in cuda.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value.cpu(), saved[name], rtol=0, atol=0,
                                       msg=f"the operator modified {name}")

    ref_time = compute_ref_performance(cuda)
    res_time = compute_res_performance(cuda)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
