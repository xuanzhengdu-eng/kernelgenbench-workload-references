# Synthetic workload; no production capture or production frequency claim.
"""OP-c4159a95fc8b: DeepSpeed fused weighted restore forward.

Inputs: combined_rows is [num_tokens*top_k, hidden] in the sorted routed order,
top_scores is fp32 [num_tokens, top_k] and token_indices_sorted is a complete
permutation of the routed rows. The output is [batches, seqlen, hidden] in the
combined_rows dtype, accumulated in fp32. This item is the DeepSpeed branch
only; the vLLM modes are separate items.
"""
import pytest
import torch
import triton.testing
import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult

COMBOS = [
    (2, 3, 2, 16, torch.bfloat16),
    (1, 3, 3, 24, torch.float16),
    (1, 2, 4, 8, torch.bfloat16),
    (2, 2, 2, 32, torch.float32),
]


def input_build(config):
    batches, seqlen, top_k, hidden, dtype = config
    num_tokens = batches * seqlen
    generator = torch.Generator(device="cpu").manual_seed(451)
    combined_rows = torch.randn((num_tokens * top_k, hidden), dtype=dtype,
                                generator=generator)
    top_scores = torch.rand((num_tokens, top_k), generator=generator)
    token_indices_sorted = torch.randperm(num_tokens * top_k, generator=generator).to(torch.int32)
    return {"combined_rows": combined_rows, "top_scores": top_scores,
            "token_indices_sorted": token_indices_sorted, "top_k": top_k,
            "shape": (batches, seqlen, hidden)}


def _cuda(inputs):
    return {k: (v.to(kernelgenbench.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_weighted_restore(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_weighted_restore(**inputs)


def _tol(dtype):
    return {torch.float32: (2e-6, 2e-6), torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (8e-3, 8e-3)}[dtype]


def compare(ref_out, res_out, dtype, shape):
    assert torch.is_tensor(res_out)
    assert res_out.shape == ref_out.shape == shape
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == ref_out.device
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_ds_weighted_restore")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_weighted_restore(config):
    dtype, shape = config[4], (config[0], config[1], config[3])
    inputs = input_build(config)
    ref_inputs = _cuda(inputs)
    res_inputs = _cuda(inputs)
    before = {name: value.clone() for name, value in res_inputs.items()
              if torch.is_tensor(value)}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype, shape)
    for name, value in before.items():
        torch.testing.assert_close(res_inputs[name], value, rtol=0, atol=0)
    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
