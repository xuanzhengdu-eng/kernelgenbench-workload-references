"""Accuracy and performance for v1_vocab_entropy_back.

Backward of DeepSpeed's vocabulary-sharded entropy.  ``logits`` is
``(rows, vocab)`` or ``(batch, time, vocab)`` in fp32/fp16/bf16; ``global_max``,
``log_sum_exp`` and ``entropy`` are the fp32 saved statistics and
``grad_output`` is the per-row upstream gradient.  The output has the logits'
shape and dtype and equals ``-grad_output * p * (log p + H)`` where ``p`` is
reconstructed from the saved statistics.

Key assertions: masked ``-inf`` logits contribute exactly zero (no NaN), an
all ``-inf`` row gets an all-zero gradient, the statistics are not modified, and
the fp32 result matches an independent fp64 evaluation.
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
    rows[0, :] = float("-inf")
    if rows.shape[0] > 2:
        rows[2, 0] = float("-inf")
    x = rows.float()
    global_max = x.amax(dim=-1)
    finite = torch.where(global_max == float("-inf"), torch.zeros_like(global_max), global_max)
    sum_exp = torch.where(global_max == float("-inf"), torch.zeros_like(global_max),
                          torch.exp(x - finite[:, None]).sum(dim=-1))
    log_sum_exp = sum_exp.log()
    logp = x - global_max[:, None] - log_sum_exp[:, None]
    entropy = -(torch.exp(logp) * logp).masked_fill(logp == float("-inf"), 0.0).sum(dim=-1)
    # An all -inf row has no distribution: its saved entropy is NaN upstream.
    grad_output = torch.linspace(0.25, 1.0, rows.shape[0])
    return {"logits": logits, "global_max": global_max, "log_sum_exp": log_sum_exp,
            "entropy": entropy, "grad_output": grad_output}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vocab_entropy_back(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vocab_entropy_back(**inputs)


def compare(ref_out, res_out, dtype):
    assert ref_out.shape == res_out.shape and ref_out.dtype == res_out.dtype == dtype
    assert not torch.isnan(res_out).any(), "solution produced NaN"
    tolerance = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(res_out, ref_out, atol=tolerance, rtol=tolerance, equal_nan=True)


def compute_ref_performance(inputs):
    """Time the reference entry on the prepared inputs (read-only operator)."""
    return measure_cuda(lambda: compute_ref(**inputs))


def compute_res_performance(inputs):
    """Time the solution entry on the same prepared inputs."""
    return measure_cuda(lambda: compute_res(**inputs))


@label("v1_vocab_entropy_back")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vocab_entropy_back(config):
    shape, dtype = config
    inputs = input_build(config)
    saved = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    cuda = {name: (value.cuda() if isinstance(value, torch.Tensor) else value)
            for name, value in inputs.items()}

    ref_out = compute_ref(**cuda)
    res_out = compute_res(**cuda)
    compare(ref_out, res_out, dtype)

    # Independent fp64 evaluation of -grad_output * p * (log p + H).
    rows = shape[-1]
    x = cuda["logits"].reshape(-1, rows).double()
    maximum = cuda["global_max"].double()
    undefined = maximum == float("-inf")
    safe_max = torch.where(undefined, torch.zeros_like(maximum), maximum)
    p = torch.exp(x - safe_max[:, None] - cuda["log_sum_exp"].double()[:, None])
    p = torch.where(torch.isneginf(x), torch.zeros_like(p), p)
    entropy = torch.where(undefined, torch.zeros_like(maximum), cuda["entropy"].double())
    expected = -(p * (torch.log(p.clamp_min(torch.finfo(torch.float64).tiny)) + entropy[:, None])) \
        * cuda["grad_output"].double()[:, None]
    expected = torch.where(undefined[:, None], torch.zeros_like(expected), expected)
    expected = expected.reshape(shape)
    tolerance = {torch.float32: 1e-5, torch.float16: 3e-3, torch.bfloat16: 3e-2}[dtype]
    torch.testing.assert_close(res_out.double(), expected, atol=tolerance, rtol=tolerance,
                               msg="entropy backward differs from the analytic gradient")
    # The all -inf row has an exactly zero gradient.
    assert torch.equal(res_out.reshape(-1, rows)[0], torch.zeros(rows, dtype=dtype, device="cuda"))
    for name, value in cuda.items():
        if isinstance(value, torch.Tensor):
            # The saved entropy of an undefined row is NaN upstream; compare with NaN equality.
            torch.testing.assert_close(value.cpu(), saved[name], rtol=0, atol=0, equal_nan=True,
                                       msg=f"the operator modified {name}")

    ref_time = compute_ref_performance(cuda)
    res_time = compute_res_performance(cuda)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
