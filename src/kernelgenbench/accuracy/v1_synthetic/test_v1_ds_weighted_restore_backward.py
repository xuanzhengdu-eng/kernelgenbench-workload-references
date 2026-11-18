# Synthetic workload; no production capture or production frequency claim.
"""OP-24a15a06532c: DeepSpeed fused weighted restore backward (explicit helper).

Inputs: grad_output [num_tokens, hidden], combined_rows [num_tokens*top_k,
hidden], top_scores [num_tokens, top_k] fp32, and a complete
token_indices_sorted permutation. Expected tuple: grad_rows, grad_scores, None,
None. grad_rows is checked for shape/dtype and fp32 values; grad_scores must
stay fp32. Scores are never recomputed from a different permutation.
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
    grad_output = torch.randn((num_tokens, hidden), dtype=dtype, generator=generator)
    return {"grad_output": grad_output, "combined_rows": combined_rows,
            "top_scores": top_scores, "token_indices_sorted": token_indices_sorted,
            "top_k": top_k}


def _cuda(inputs):
    return {k: (v.to(kernelgenbench.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_weighted_restore_backward(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_weighted_restore_backward(**inputs)


def _tol(dtype):
    return {torch.float32: (2e-6, 2e-6), torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (8e-3, 8e-3)}[dtype]


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and len(ref_out) == 4
    assert isinstance(res_out, tuple) and len(res_out) == 4
    assert ref_out[2] is None and ref_out[3] is None
    assert res_out[2] is None and res_out[3] is None
    rtol, atol = _tol(dtype)
    for index in (0, 1):
        assert res_out[index].shape == ref_out[index].shape
        assert res_out[index].dtype == ref_out[index].dtype
        assert res_out[index].device == ref_out[index].device
        torch.testing.assert_close(res_out[index], ref_out[index], rtol=rtol, atol=atol)
    assert ref_out[0].dtype == dtype
    assert ref_out[1].dtype == torch.float32


def compute_ref_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_ds_weighted_restore_backward")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_weighted_restore_backward(config):
    dtype = config[4]
    inputs = input_build(config)
    ref_inputs = _cuda(inputs)
    res_inputs = _cuda(inputs)
    before = {name: value.clone() for name, value in res_inputs.items()
              if torch.is_tensor(value)}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    for name, value in before.items():
        torch.testing.assert_close(res_inputs[name], value, rtol=0, atol=0)
    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
