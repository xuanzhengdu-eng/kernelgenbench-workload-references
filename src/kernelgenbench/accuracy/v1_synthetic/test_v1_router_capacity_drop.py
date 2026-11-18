# Synthetic workload; no production capture or production frequency claim.
"""OP-5b72121cfb1a, official sub-contract: Megatron token dropping with probability or position policy and optional capacity padding (NVIDIA/Megatron-LM@266f1c97dca477ca8b92c16087506da2000b0b84, megatron/core/transformer/moe/moe_utils.py:877-940). Position policy ranks a 0/1 tensor, so tied selections may vary; its test checks the capacity/subset properties instead of assuming an order. Synthetic FP32 probabilities are tie-free at the top score. Other dtypes remain unverified."""
import math

import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.synthetic_inputs import build_case
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# routing_probs=routing_map=(B*T,16) float32/bool, router_topk=4.
# capacity_factor=0.5 keeps capacity below the token count, 5.0 forces the
# "capacity exceeds tokens, keep everything" branch.
# (shape, dtype, drop_policy, pad_to_capacity, capacity_factor)
COMBOS = [(shape, dtype, policy, pad, factor)
          for shape in [(2, 16, 128), (2, 128, 1024)]
          for dtype in (torch.float32,)
          for policy in ('probs', 'position')
          for pad in (False, True)
          for factor in (0.5, 5.0)]


def input_build(config):
    """Build the routing tensors on CPU; rows are (B*T, 16)."""
    shape, dtype, policy, pad, factor = config
    inputs = build_case('capacity_drop', shape, dtype, 'cpu')
    inputs.update(capacity_factor=factor, pad_to_capacity=pad, drop_policy=policy)
    return inputs


def to_device(inputs):
    return {k: (v.to(kernelgenbench.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()}


def compute_ref(routing_probs, routing_map, router_topk, capacity_factor,
                pad_to_capacity, drop_policy):
    return kernelgenbench.baseline.v1_router_capacity_drop(
        routing_probs, routing_map, router_topk, capacity_factor,
        pad_to_capacity, drop_policy)


def compute_res(routing_probs, routing_map, router_topk, capacity_factor,
                pad_to_capacity, drop_policy):
    return kernelgenbench.solution.v1_router_capacity_drop(
        routing_probs, routing_map, router_topk, capacity_factor,
        pad_to_capacity, drop_policy)


def compare(ref_out, res_out, dtype, policy='probs', inputs=None):
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    if policy == 'probs':
        # Top scores are tie-free, so the kept set is fully determined.
        for got, want, name in ((res_out[0], ref_out[0], 'final_probs'),
                                (res_out[1], ref_out[1], 'final_map')):
            assert got.shape == want.shape and got.device == want.device
            assert got.dtype == want.dtype
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        return
    # Position policy: topk over 0/1 ties freely, so assert the contract.
    probs, chosen = res_out
    route = inputs['routing_map']
    raw = inputs['routing_probs']
    assert chosen.shape == route.shape and chosen.device == route.device
    assert chosen.dtype == torch.bool
    cap = math.ceil(route.shape[0] * inputs['router_topk'] / route.shape[1]
                    * inputs['capacity_factor'])
    kept = (chosen & route).sum(0)
    available = route.sum(0)
    assert torch.equal(kept, available.clamp(max=cap)), 'incorrect selected-token count'
    if inputs['pad_to_capacity']:
        assert (chosen.sum(0) == min(cap, route.shape[0])).all(), 'incorrect capacity padding'
    else:
        assert not (chosen & ~route).any(), 'unselected tokens included'
    assert probs.dtype == raw.dtype and probs.shape == raw.shape
    torch.testing.assert_close(probs, raw * chosen, rtol=0, atol=0)


def measure(fn, inputs):
    import triton.testing
    return triton.testing.do_bench(lambda: fn(**inputs), warmup=25, rep=100)


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure(compute_res, inputs)


@label('v1_router_capacity_drop')
@parametrize('config', COMBOS)
@pytest.mark.parametrize('config', COMBOS)
def test_accuracy_v1_router_capacity_drop(config):
    shape, dtype, policy, pad, factor = config
    inputs = to_device(input_build(config))
    ref_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    res_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    expected = compute_ref(**ref_inputs)
    actual = compute_res(**res_inputs)
    compare(expected, actual, dtype, policy=policy, inputs=res_inputs)
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, res_inputs[key]), f'candidate modified input: {key}'
    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float('inf'))
