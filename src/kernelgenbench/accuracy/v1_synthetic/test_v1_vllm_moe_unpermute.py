# Synthetic workload; no production capture or production frequency claim.
"""OP-c4159a95fc8b (split branch): vLLM moe_unpermute.

Inputs: permuted_hidden_states is [num_tokens*topk, hidden] in the token dtype,
topk_weights is fp32 [num_tokens, topk] and inv_permuted_idx is int32
[num_tokens, topk]. When expert_first_token_offset is given its last element
bounds the valid rows. The native CUDA kernel overwrites `out` and returns
None; the assertions cover the in-place result, the None return and untouched
inputs.
"""
import pytest
import torch
import triton.testing
import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult

# num_tokens, topk, hidden, dtype, use expert offset
COMBOS = [
    (3, 2, 16, torch.float16, False),
    (4, 3, 64, torch.bfloat16, True),
    (2, 4, 32, torch.float16, True),
    (5, 2, 64, torch.bfloat16, False),
]


def input_build(config):
    num_tokens, topk, hidden, dtype, use_offset = config
    generator = torch.Generator(device="cpu").manual_seed(501 + num_tokens + topk)
    rows = torch.randn((num_tokens * topk, hidden), dtype=dtype, generator=generator)
    weights = torch.rand((num_tokens, topk), generator=generator)
    inv = torch.randperm(num_tokens * topk, generator=generator).to(torch.int32)
    inv = inv.reshape(num_tokens, topk).contiguous()
    offset = None
    if use_offset:
        offset = torch.tensor([0, num_tokens * topk - 2], dtype=torch.int64)
    return {"out": torch.full((num_tokens, hidden), -55.0, dtype=dtype),
            "permuted_hidden_states": rows, "topk_weights": weights,
            "inv_permuted_idx": inv, "expert_first_token_offset": offset}


def _cuda(inputs):
    result = {}
    for name, value in inputs.items():
        result[name] = value.to(kernelgenbench.device) if torch.is_tensor(value) else value
    return result


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_moe_unpermute(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_moe_unpermute(**inputs)


def _tol(dtype):
    return {torch.float16: (1e-3, 1e-3), torch.bfloat16: (8e-3, 8e-3),
            torch.float32: (2e-6, 2e-6)}[dtype]


def compare(ref_out, res_out, dtype=None):
    # Both the reference and the official kernel write their result buffer and
    # return None; the buffer contents are compared by the caller.
    assert ref_out is None and res_out is None, "official kernel writes the buffer and returns None"


def compute_ref_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_vllm_moe_unpermute")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_moe_unpermute(config):
    dtype = config[3]
    inputs = input_build(config)
    ref_inputs = _cuda(inputs)
    res_inputs = _cuda(inputs)
    before = {name: value.clone() for name, value in res_inputs.items()
              if torch.is_tensor(value) and name != "out"}
    ptr = res_inputs["out"].data_ptr()
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    assert res_inputs["out"].data_ptr() == ptr
    rtol, atol = _tol(dtype)
    assert res_inputs["out"].shape == ref_inputs["out"].shape
    torch.testing.assert_close(res_inputs["out"], ref_inputs["out"], rtol=rtol, atol=atol)
    for name, value in before.items():
        torch.testing.assert_close(res_inputs[name], value, rtol=0, atol=0)
    ref_time = compute_ref_performance(**{**ref_inputs, "out": inputs["out"].to(kernelgenbench.device)})
    res_time = compute_res_performance(**{**res_inputs, "out": inputs["out"].to(kernelgenbench.device)})
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
