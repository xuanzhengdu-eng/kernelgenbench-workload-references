"""Accuracy and performance for v1_vocab_logprobs_back.

Backward of DeepSpeed's vocabulary-sharded selected log-probability.  Inputs:
``logits`` is ``(rows, vocab)`` or ``(batch, time, vocab)`` in fp32/fp16/bf16,
``index`` is ``(rows, 3)`` int64 with repeated ids and ``ignore_index=-100``,
``global_max``/``log_sum_exp`` are the fp32 saved statistics and ``grad_output``
is the upstream gradient on the selected ids.  The output has the logits' shape
and dtype.

Key assertions: rows whose ids are all ignored have no softmax term (only the
one-hot terms survive); a row gradient sums to the upstream sum because the
softmax term cancels; logits and index are not modified.
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
    global_max, log_sum_exp = _stats(rows.float())
    grad_output = torch.linspace(0.1, 1.0, index.numel()).reshape_as(index)
    return {"logits": logits, "global_max": global_max, "log_sum_exp": log_sum_exp,
            "index": index, "ignore_index": -100, "grad_output": grad_output}


def _stats(x):
    """fp32 per-row (max, shifted log-sum-exp); an all -inf row has no softmax."""
    global_max = x.amax(dim=-1)
    finite_max = torch.where(global_max == float("-inf"), torch.zeros_like(global_max), global_max)
    sum_exp = torch.where(global_max == float("-inf"), torch.zeros_like(global_max),
                          torch.exp(x - finite_max[:, None]).sum(dim=-1))
    return global_max, sum_exp.log()


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vocab_logprobs_back(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vocab_logprobs_back(**inputs)


def compare(ref_out, res_out, dtype):
    assert ref_out.shape == res_out.shape and ref_out.dtype == res_out.dtype == dtype
    tolerance = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(res_out, ref_out, atol=tolerance, rtol=tolerance, equal_nan=True)


def compute_ref_performance(inputs):
    """Time the reference entry on the prepared inputs (read-only operator)."""
    return measure_cuda(lambda: compute_ref(**inputs))


def compute_res_performance(inputs):
    """Time the solution entry on the same prepared inputs."""
    return measure_cuda(lambda: compute_res(**inputs))


@label("v1_vocab_logprobs_back")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vocab_logprobs_back(config):
    shape, dtype = config
    inputs = input_build(config)
    saved = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    cuda = {name: (value.cuda() if isinstance(value, torch.Tensor) else value)
            for name, value in inputs.items()}
    # The supplied statistics must be the ones the official forward recomputes.
    x = cuda["logits"].reshape(-1, shape[-1]).float()
    expected_max, expected_logsum = _stats(x)
    torch.testing.assert_close(cuda["global_max"].cuda(), expected_max.cuda(), rtol=0, atol=0)
    torch.testing.assert_close(cuda["log_sum_exp"].cuda(), expected_logsum.cuda(), rtol=1e-6, atol=1e-6)

    ref_out = compute_ref(**cuda)
    res_out = compute_res(**cuda)
    compare(ref_out, res_out, dtype)

    rows = shape[-1]
    flat_ref = ref_out.reshape(-1, rows)
    ignore = -100
    valid = cuda["index"] != ignore
    upstream = cuda["grad_output"].float() * valid
    sum_tolerance = {torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 3e-2}[dtype]
    for row in range(flat_ref.shape[0]):
        if cuda["global_max"][row] == float("-inf"):
            # No softmax term: only the one-hot terms remain.
            expected_total = upstream[row].sum()
        else:
            # The softmax term cancels the one-hot terms.
            expected_total = torch.zeros((), device="cuda")
        torch.testing.assert_close(flat_ref[row].float().sum(), expected_total.float(),
                                   rtol=sum_tolerance, atol=sum_tolerance,
                                   msg=f"row {row} gradient sum")
    for name, value in cuda.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value.cpu(), saved[name], rtol=0, atol=0,
                                       msg=f"solution or reference modified {name}")

    ref_time = compute_ref_performance(cuda)
    res_time = compute_res_performance(cuda)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
