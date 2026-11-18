# Synthetic workload; no production capture or production frequency claim.
"""OP-81914edec43a, official sub-contract: Megatron unfused MoE auxiliary routing scores and boolean expert map with optional mask (NVIDIA/Megatron-LM@266f1c97dca477ca8b92c16087506da2000b0b84, megatron/core/transformer/moe/moe_utils.py:826-874). The official docstring calls the mask a validity mask, but the code multiplies by (~padding_mask), i.e. True means padding; this test pins the code behaviour. Only the official softmax and sigmoid score functions exist at the fixed commit. Token-count collectives and the TE fused path are not covered."""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.synthetic_inputs import build_case
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# logits=(B*T,16) with strictly ordered rows so topk is tie-free, topk=4,
# padding_mask=(B*T,) bool where True marks a padding row.
# (shape, dtype, score_function, masked)
COMBOS = [(shape, dtype, score_function, masked)
          for shape in [(2, 16, 128), (2, 128, 1024)]
          for dtype in (torch.float32, torch.float16, torch.bfloat16)
          for score_function in ('softmax', 'sigmoid')
          for masked in (False, True)]


def input_build(config):
    shape, dtype, score_function, masked = config
    inputs = build_case('aux_router', shape, dtype, 'cpu')
    inputs['score_function'] = score_function
    if masked:
        # Every fifth token is padding.
        inputs['padding_mask'] = torch.arange(shape[0] * shape[1]) % 5 == 0
    return inputs


def to_device(inputs):
    return {k: (v.to(kernelgenbench.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()}


def compute_ref(logits, topk, score_function, padding_mask):
    return kernelgenbench.baseline.v1_aux_router_scores(
        logits, topk, score_function, padding_mask)


def compute_res(logits, topk, score_function, padding_mask):
    return kernelgenbench.solution.v1_aux_router_scores(
        logits, topk, score_function, padding_mask)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    map_ref, scores_ref = ref_out
    map_res, scores_res = res_out
    # Boolean routing map: exact equality.
    assert map_res.shape == map_ref.shape and map_res.device == map_ref.device
    assert map_res.dtype == torch.bool and map_ref.dtype == torch.bool
    assert torch.equal(map_res, map_ref), 'routing map differs'
    assert map_res.sum(dim=1).max().item() <= 4
    # Scores stay normalized float32 regardless of the logits dtype.
    assert scores_res.shape == scores_ref.shape
    assert scores_res.device == scores_ref.device
    assert scores_res.dtype == scores_ref.dtype == torch.float32
    tol = 2e-5 if dtype == torch.float32 else 1e-3
    torch.testing.assert_close(scores_res, scores_ref, rtol=tol, atol=tol)


def measure(fn, inputs):
    import triton.testing
    return triton.testing.do_bench(lambda: fn(**inputs), warmup=25, rep=100)


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure(compute_res, inputs)


@label('v1_aux_router_scores')
@parametrize('config', COMBOS)
@pytest.mark.parametrize('config', COMBOS)
def test_accuracy_v1_aux_router_scores(config):
    dtype = config[1]
    inputs = to_device(input_build(config))
    ref_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    res_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    expected = compute_ref(**ref_inputs)
    actual = compute_res(**res_inputs)
    compare(expected, actual, dtype)
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, res_inputs[key]), f'candidate modified input: {key}'
    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float('inf'))
