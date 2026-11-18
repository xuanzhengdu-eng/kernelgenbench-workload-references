"""Accuracy and performance for v1_log_softmax (723 row 53).

Official entry: Megatron ``batch_invariant_kernels.log_softmax`` (Triton
``_log_softmax_kernel``, one program per row, fp32 max/exp accumulation).
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/custom_layers/batch_invariant_kernels.py#L308-L340

Inputs: ``input`` is any-rank (leading dims are flattened to rows) with the
softmax over the last axis; ``dim`` is -1 or the last axis. Key assertions: the
output keeps shape and dtype, each row's ``exp`` sums to one, the input is not
modified, and a non-last ``dim`` raises ``ValueError`` on both sides.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows52_61_checks as checks
from sandbox.verifier.test_parametrize import label, parametrize


# (shape, dtype): small rows, a 4096-wide row, and a 1-D vector.
COMBOS = [
    (shape, dtype)
    for shape in [(2, 16, 128), (2, 128, 1024), (4, 4096), (3,)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]

TOLERANCE = {
    torch.float32: (1e-5, 1e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}


def input_build(config):
    shape, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    x = torch.randn(shape, dtype=torch.float32, generator=generator) * 2.0
    return {"input": x.to(dtype), "dim": -1}


def compute_ref(input, dim):
    return kernelgenbench.baseline.v1_log_softmax(input, dim)


def compute_res(input, dim):
    return kernelgenbench.solution.v1_log_softmax(input, dim)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape, (res_out.shape, ref_out.shape)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == ref_out.device
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out.float(), ref_out.float(), rtol=rtol, atol=atol)
    # log_softmax invariant: exp of every row sums to one. The output is stored
    # in the input dtype, so the row sum inherits that storage rounding
    # (a few 1e-3 in fp16, up to ~2e-2 in bf16).
    total = res_out.float().exp().sum(dim=-1)
    invariant_tol = {torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(total, torch.ones_like(total), rtol=invariant_tol,
                               atol=invariant_tol)


def compute_ref_performance(input, dim):
    return checks.measure(kernelgenbench.baseline.v1_log_softmax,
                          {"input": input, "dim": dim},
                          kernelgenbench.device)


def compute_res_performance(input, dim):
    return checks.measure(kernelgenbench.solution.v1_log_softmax,
                          {"input": input, "dim": dim},
                          kernelgenbench.device)


@label("v1_log_softmax")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_log_softmax(config):
    shape, dtype = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    assert res_inputs["input"].shape == shape
    assert res_inputs["input"].dtype == dtype

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    assert not torch._C._is_alias_of(res_out, res_inputs["input"]), "output aliases input"
    torch.testing.assert_close(res_inputs["input"].cpu(), inputs["input"], rtol=0, atol=0)

    # Only the last dimension is supported; both sides reject dim=0 for rank > 1.
    if len(shape) > 1:
        for call in (compute_ref, compute_res):
            try:
                call(ref_inputs["input"], 0)
            except ValueError:
                pass
            else:
                raise AssertionError("a non-last dim must raise ValueError")

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
