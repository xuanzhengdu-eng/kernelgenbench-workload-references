# Synthetic workload; no production capture or production frequency claim.
"""OP-c4159a95fc8b (split branch): vLLM DeepGEMM unpermute and reduce.

Inputs: `a` is [num_tokens*topk, hidden] with a padded row stride, topk_ids and
inv_perm are int32 [num_tokens, topk], topk_weights is fp32 and the optional
expert_map remaps global expert ids (mapped -1 drops the slot). The Triton
ep_gather kernel writes `output` in place and returns None. The assertions cover
the strided input read, the padded columns left untouched, and the None return.
"""
import pytest
import torch
import triton.testing
import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult

# num_tokens, topk, hidden, dtype, use expert map, input row padding
COMBOS = [
    (3, 2, 16, torch.float16, False, 0),
    (4, 3, 64, torch.bfloat16, True, 7),
    (2, 4, 32, torch.float16, True, 3),
    (5, 2, 64, torch.bfloat16, False, 5),
]


def input_build(config):
    num_tokens, topk, hidden, dtype, use_map, pad = config
    generator = torch.Generator(device="cpu").manual_seed(455 + num_tokens + topk)
    base = torch.randn((num_tokens * topk, hidden + pad), generator=generator).to(dtype)
    a = base[:, :hidden]
    weights = torch.rand((num_tokens, topk), generator=generator)
    topk_ids = (torch.arange(num_tokens * topk, dtype=torch.int32).reshape(num_tokens, topk) % 4)
    topk_ids[0, 0] = -1
    inv = torch.randperm(num_tokens * topk, generator=generator).reshape(num_tokens, topk)
    inv = inv.to(torch.int32).contiguous()
    inv[0, 0] = -77  # never read because topk_ids[0, 0] == -1
    expert_map = torch.tensor([0, -1, 1, 2], dtype=torch.int32) if use_map else None
    return {"a": a, "topk_ids": topk_ids.contiguous(), "topk_weights": weights,
            "inv_perm": inv, "expert_map": expert_map,
            "output": torch.full((num_tokens, hidden), -55.0, dtype=dtype),
            "base": base}


def _cuda(inputs):
    result = {}
    base = inputs["base"].to(kernelgenbench.device)
    for name, value in inputs.items():
        if name == "a":
            continue  # rebuilt as a strided view of the moved base buffer
        result[name] = value.to(kernelgenbench.device) if torch.is_tensor(value) else value
    result["base"] = base
    result["a"] = base[:, :inputs["a"].shape[1]]
    return result


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_deepgemm_unpermute_and_reduce(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_deepgemm_unpermute_and_reduce(**inputs)


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


@label("v1_vllm_deepgemm_unpermute_and_reduce")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_deepgemm_unpermute_and_reduce(config):
    dtype, pad = config[3], config[5]
    inputs = input_build(config)
    ref_inputs = _cuda(inputs)
    res_inputs = _cuda(inputs)
    ref_base_before = ref_inputs["base"].clone()
    res_base_before = res_inputs["base"].clone()
    before = {name: value.clone() for name, value in res_inputs.items()
              if torch.is_tensor(value) and name not in ("output", "base")}
    ptr = res_inputs["output"].data_ptr()
    ref_call = {k: v for k, v in ref_inputs.items() if k != "base"}
    res_call = {k: v for k, v in res_inputs.items() if k != "base"}
    ref_out = compute_ref(**ref_call)
    res_out = compute_res(**res_call)
    compare(ref_out, res_out, dtype)
    assert res_inputs["output"].data_ptr() == ptr
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(res_inputs["output"], ref_inputs["output"], rtol=rtol, atol=atol)
    if pad:
        assert torch.equal(res_inputs["base"][:, inputs["a"].shape[1]:],
                           res_base_before[:, inputs["a"].shape[1]:])
        assert torch.equal(ref_inputs["base"][:, inputs["a"].shape[1]:],
                           ref_base_before[:, inputs["a"].shape[1]:])
    for name, value in before.items():
        torch.testing.assert_close(res_inputs[name], value, rtol=0, atol=0)
    ref_time = compute_ref_performance(**ref_call)
    res_time = compute_res_performance(**res_call)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
