# Synthetic workload; no production capture or production frequency claim.
"""OP-c4159a95fc8b: Megatron training MoE unpermute, non-fused path.

Inputs: permuted_tokens [num_routed, 8] in expert-major order, sorted_indices
[num_routed] and restore_shape (4, 8); probs/routing_map are optional. Expected
output is the [4, 8] restoration, weighted by the routed probabilities when
given. A scalar Python loop rebuilds the expected scatter-add, so the test does
not rely only on the reference implementation.
"""
import pytest
import torch
import triton.testing
import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult

# drop_and_pad, has_probs, dtype
COMBOS = [
    (False, False, torch.float32),
    (False, True, torch.float32),
    (False, True, torch.bfloat16),
    (True, True, torch.float32),
    (True, False, torch.bfloat16),
]

DROP_ROUTE = [[1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0]]
KEEP_ROUTE = [[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1]]


def input_build(config):
    drop, has_probs, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(911)
    tokens = torch.randn((4, 8), dtype=dtype, generator=generator)
    route = torch.tensor(DROP_ROUTE if drop else KEEP_ROUTE, dtype=torch.bool)
    probs = torch.rand((4, 3), generator=generator) if has_probs else None
    selected = torch.nonzero(route.T.contiguous().reshape(-1), as_tuple=False).flatten()
    ids = selected % 4
    return {"permuted_tokens": tokens.index_select(0, ids), "sorted_indices": ids,
            "restore_shape": (4, 8), "probs": probs, "routing_map": route,
            "drop_and_pad": drop}


def _cuda(inputs):
    return {k: (v.to(kernelgenbench.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_moe_unpermute(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_moe_unpermute(**inputs)


def _tol(dtype):
    return {torch.float32: (2e-6, 2e-6), torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (8e-3, 8e-3)}[dtype]


def _scalar_oracle(permuted, ids, probs, route, drop, restore_shape):
    num_tokens, num_experts = route.shape
    if probs is None:
        weights = None
    elif drop:
        capacity = ids.numel() // num_experts
        picked = []
        for expert in range(num_experts):
            for slot in range(capacity):
                token = int(ids[expert * capacity + slot])
                picked.append(float(probs[token, expert]))
        weights = torch.tensor(picked, dtype=torch.float32)
    else:
        weights = probs.T.contiguous().masked_select(route.T.contiguous())
    expected = torch.zeros(restore_shape, dtype=torch.float64)
    for position in range(ids.numel()):
        token = int(ids[position])
        row = permuted[position].double()
        if weights is not None:
            row = row * float(weights[position])
        expected[token] += row
    return expected


def compare(ref_out, res_out, dtype, permuted, ids, probs, route, drop, restore_shape):
    assert torch.is_tensor(res_out)
    assert res_out.shape == ref_out.shape == restore_shape
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == ref_out.device
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)
    expected = _scalar_oracle(permuted.cpu(), ids.cpu(), None if probs is None else probs.cpu(),
                              route.cpu(), drop, restore_shape)
    torch.testing.assert_close(ref_out.cpu().float(), expected.float(), rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_megatron_moe_unpermute")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_moe_unpermute(config):
    dtype, drop = config[2], config[0]
    inputs = input_build(config)
    ref_inputs = _cuda(inputs)
    res_inputs = _cuda(inputs)
    before = {name: value.clone() for name, value in res_inputs.items()
              if torch.is_tensor(value)}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype, res_inputs["permuted_tokens"],
            res_inputs["sorted_indices"], res_inputs["probs"],
            res_inputs["routing_map"], drop, res_inputs["restore_shape"])
    for name, value in before.items():
        torch.testing.assert_close(res_inputs[name], value, rtol=0, atol=0)
    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
