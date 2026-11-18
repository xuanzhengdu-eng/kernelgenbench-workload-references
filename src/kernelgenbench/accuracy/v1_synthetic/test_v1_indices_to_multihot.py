# Synthetic workload; no production capture or production frequency claim.
"""OP-abd21066927b: Megatron IndicesToMultihot forward.

Inputs: indices and probs_indices are [num_tokens, 3] int32 / float with
topk=3 and num_of_local_experts=16. Every row's ids are unique; -1 and ids
>= 16 are the ignored padding values. The key assertions compare the bool
multihot and the gathered probabilities, and check that neither input was
modified. Timing uses the same CUDA inputs on both sides.
"""
import pytest
import torch
import triton.testing
import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult

NUM_EXPERTS = 16
TOPK = 3
COMBOS = [(num_tokens, dtype)
          for num_tokens in (32, 256)
          for dtype in (torch.float32, torch.float16, torch.bfloat16)]


def input_build(config):
    num_tokens, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    base = torch.arange(num_tokens, dtype=torch.int32).unsqueeze(1)
    indices = (base + torch.arange(TOPK, dtype=torch.int32).unsqueeze(0)) % NUM_EXPERTS
    indices[::4, 0] = -1              # ignored padding id
    indices[1::4, 1] = NUM_EXPERTS + 2  # ignored out-of-range id
    probs = torch.randn((num_tokens, TOPK), dtype=dtype, generator=generator)
    return {"indices": indices.contiguous(), "probs_indices": probs,
            "num_of_local_experts": NUM_EXPERTS}


def _cuda(inputs):
    return {k: (v.to(kernelgenbench.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_indices_to_multihot(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_indices_to_multihot(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    ref_present, ref_probs = ref_out
    res_present, res_probs = res_out
    assert res_present.dtype == ref_present.dtype == torch.bool
    assert res_probs.dtype == ref_probs.dtype == dtype
    assert res_present.shape == ref_present.shape == (res_probs.shape[0], NUM_EXPERTS)
    assert torch.equal(res_present, ref_present), "multihot flags differ"
    torch.testing.assert_close(res_probs, ref_probs, rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_indices_to_multihot")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_indices_to_multihot(config):
    dtype = config[1]
    inputs = input_build(config)
    ref_inputs = _cuda(inputs)
    res_inputs = _cuda(inputs)
    ref_before = {k: v.clone() for k, v in ref_inputs.items() if torch.is_tensor(v)}
    res_before = {k: v.clone() for k, v in res_inputs.items() if torch.is_tensor(v)}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    for before, side in ((ref_before, ref_inputs), (res_before, res_inputs)):
        for name, value in before.items():
            torch.testing.assert_close(side[name], value, rtol=0, atol=0)
    ref_time = compute_ref_performance(**{k: v for k, v in ref_inputs.items()})
    res_time = compute_res_performance(**{k: v for k, v in res_inputs.items()})
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
