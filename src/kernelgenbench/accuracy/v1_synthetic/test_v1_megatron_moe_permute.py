# Synthetic workload; no production capture or production frequency claim.
"""OP-c0f3b4fcda53: Megatron training MoE permute, non-fused path.

Inputs: tokens [4, 8] with a boolean routing_map [4, 3]; `num_out_tokens` is the
routed-pair count (or a smaller truncation), probs is optional fp32 [4, 3] and
tokens_per_expert is passed through unchanged. The expected tuple is
(tokens, probs, sorted_indices, None, tokens_per_expert); the independent oracle
rebuilds the expert-major order from first principles. The fused Transformer
Engine path is not part of this item.
"""
import pytest
import torch
import triton.testing
import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult

# drop_and_pad, has_probs, dtype, num_out_tokens (None means every routed pair)
COMBOS = [
    (False, False, torch.float32, None),
    (False, True, torch.float32, None),
    (False, True, torch.bfloat16, None),
    (True, True, torch.float32, None),
    (True, False, torch.bfloat16, None),
    (False, True, torch.float32, 6),
]

DROP_ROUTE = [[1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0]]
KEEP_ROUTE = [[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1]]


def input_build(config):
    drop, has_probs, dtype, override = config
    generator = torch.Generator(device="cpu").manual_seed(911)
    tokens = torch.randn((4, 8), dtype=dtype, generator=generator)
    route = torch.tensor(DROP_ROUTE if drop else KEEP_ROUTE, dtype=torch.bool)
    probs = torch.rand((4, 3), generator=generator) if has_probs else None
    routed = int(route.sum())
    num_out = routed if override is None else override
    return {"tokens": tokens, "routing_map": route, "probs": probs,
            "num_out_tokens": num_out, "drop_and_pad": drop,
            "tokens_per_expert": route.sum(0).to(torch.int64)}


def _cuda(inputs):
    return {k: (v.to(kernelgenbench.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_moe_permute(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_moe_permute(**inputs)


def _expected_order(route, num_out):
    flat = torch.nonzero(route.T.contiguous().reshape(-1), as_tuple=False).flatten()[:num_out]
    return flat, flat % route.shape[0]


def _tol(dtype):
    return {torch.float32: (2e-6, 2e-6), torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (8e-3, 8e-3)}[dtype]


def compare(ref_out, res_out, dtype, tokens, route, probs, num_out):
    assert isinstance(ref_out, tuple) and len(ref_out) == 5
    assert isinstance(res_out, tuple) and len(res_out) == 5
    assert ref_out[3] is None and res_out[3] is None
    rtol, atol = _tol(dtype)
    expected_flat, expected_ids = _expected_order(route.cpu(), num_out)
    tokens_cpu, probs_cpu = tokens.cpu(), None if probs is None else probs.cpu()
    for side in (ref_out, res_out):
        assert side[0].shape == (num_out, tokens.shape[1])
        assert side[0].dtype == dtype
        torch.testing.assert_close(side[0].cpu(), tokens_cpu.index_select(0, expected_ids),
                                   rtol=0, atol=0)
        assert side[2].shape == (num_out,)
        assert torch.equal(side[2].cpu(), expected_ids)
        if probs_cpu is None:
            assert side[1] is None
        else:
            assert side[1].shape == (num_out,)
            expected_probs = probs_cpu.T.contiguous().reshape(-1).index_select(0, expected_flat)
            torch.testing.assert_close(side[1].cpu(), expected_probs, rtol=0, atol=0)
    assert res_out[0].dtype == ref_out[0].dtype == dtype
    assert torch.equal(res_out[2].cpu(), ref_out[2].cpu())


def compute_ref_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_megatron_moe_permute")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_moe_permute(config):
    dtype = config[2]
    inputs = input_build(config)
    ref_inputs = _cuda(inputs)
    res_inputs = _cuda(inputs)
    before = {name: value.clone() for name, value in res_inputs.items()
              if torch.is_tensor(value)}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype, res_inputs["tokens"], res_inputs["routing_map"],
            res_inputs["probs"], res_inputs["num_out_tokens"])
    assert ref_out[4] is ref_inputs["tokens_per_expert"], "reference must pass its input through"
    assert res_out[4] is res_inputs["tokens_per_expert"], "solution must pass its input through"
    for name, value in before.items():
        torch.testing.assert_close(res_inputs[name], value, rtol=0, atol=0)
    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
