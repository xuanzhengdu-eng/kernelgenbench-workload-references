# Synthetic workload; no production capture or production frequency claim.
"""OP-f09a2aa29254: Megatron inference local expert token count.

Inputs: routing_map is [num_tokens, topk] int32 with expert ids on both sides
of [local_expert_start, local_expert_start + num_local_experts); ids outside
that window, including negative ids, must be ignored. The expected int32
[num_local_experts] vector is also rebuilt independently with bincount in the
test, so the assertion does not rely only on the reference.
"""
import pytest
import torch
import triton.testing
import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult

COMBOS = [
    (64, 2, 8, 3),
    (128, 4, 16, 0),
    (256, 2, 4, 12),
]


def input_build(config):
    num_tokens, topk, num_local_experts, local_expert_start = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    low = local_expert_start - 2
    high = local_expert_start + num_local_experts + 2
    routing_map = torch.randint(low, high, (num_tokens, topk), dtype=torch.int32,
                                generator=generator)
    routing_map[0, 0] = local_expert_start - 1
    routing_map[0, 1] = local_expert_start + num_local_experts
    return {"routing_map": routing_map.contiguous(),
            "local_expert_start": local_expert_start,
            "num_local_experts": num_local_experts}


def _cuda(inputs):
    return {k: (v.to(kernelgenbench.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_count_local_expert_tokens(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_count_local_expert_tokens(**inputs)


def compare(ref_out, res_out, routing_map, local_expert_start, num_local_experts):
    assert res_out.dtype == ref_out.dtype == torch.int32
    assert res_out.shape == ref_out.shape == (num_local_experts,)
    assert torch.equal(res_out, ref_out), "local expert counts differ"
    local = routing_map.long() - local_expert_start
    valid = (local >= 0) & (local < num_local_experts)
    oracle = torch.bincount(local[valid].reshape(-1), minlength=num_local_experts).to(torch.int32)
    assert torch.equal(ref_out, oracle), "reference disagrees with the independent bincount oracle"
    assert int(valid.sum()) == int(ref_out.sum())


def compute_ref_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_count_local_expert_tokens")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_count_local_expert_tokens(config):
    num_tokens, topk, num_local_experts, local_expert_start = config
    inputs = input_build(config)
    ref_inputs = _cuda(inputs)
    res_inputs = _cuda(inputs)
    before = res_inputs["routing_map"].clone()
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, res_inputs["routing_map"], local_expert_start,
            num_local_experts)
    assert torch.equal(res_inputs["routing_map"], before)
    assert ref_inputs["routing_map"].shape == (num_tokens, topk)
    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
