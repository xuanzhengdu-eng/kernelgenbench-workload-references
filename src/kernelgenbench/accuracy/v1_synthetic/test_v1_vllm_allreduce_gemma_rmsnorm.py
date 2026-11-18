"""Accuracy and performance for all-reduce + Gemma residual RMSNorm."""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows42_51_test_utils as u
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# hidden_states and residual share (tokens, hidden); the returned tuple is
# (normed, hidden + residual), both in the activation dtype.  A single-rank
# process group is the identity, so this exercises the unfused collective path
# without inter-device communication (as documented in the reference).
COMBOS = [(tokens, hidden, dtype)
          for tokens, hidden in ((16, 128), (5, 64))
          for dtype in (torch.bfloat16, torch.float16)]


def input_build(config):
    tokens, hidden, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(3)
    hidden_states = torch.randn((tokens, hidden), dtype=dtype, generator=generator) * 2.0
    residual = torch.randn((tokens, hidden), dtype=dtype, generator=generator)
    weight = torch.randn((hidden,), dtype=dtype, generator=generator) * 0.5
    return {"hidden_states": hidden_states, "residual": residual,
            "weight": weight, "eps": 1e-6, "group": None}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_allreduce_gemma_rmsnorm(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_allreduce_gemma_rmsnorm(**inputs)


def compare(ref_out, res_out, tokens, hidden, dtype):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 2
    rtol, atol = u.tolerance(dtype)
    for index, name in enumerate(("normed", "residual")):
        assert res_out[index].shape == ref_out[index].shape == (tokens, hidden)
        assert res_out[index].dtype == ref_out[index].dtype == dtype, name
        torch.testing.assert_close(res_out[index], ref_out[index], rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return u.measure(compute_ref, u.fresh(inputs, kernelgenbench.device))


def compute_res_performance(**inputs):
    return u.measure(compute_res, u.fresh(inputs, kernelgenbench.device))


@label("v1_vllm_allreduce_gemma_rmsnorm")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_allreduce_gemma_rmsnorm(config):
    tokens, hidden, dtype = config
    inputs = input_build(config)
    ref_inputs = u.fresh(inputs, kernelgenbench.device)
    res_inputs = u.fresh(inputs, kernelgenbench.device)
    before = u.clone_tensors(res_inputs)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, tokens, hidden, dtype)

    # The second output is exactly hidden + residual (dtype-cast), and the
    # first is the Gemma RMSNorm of it with the additive (1 + weight) offset.
    summed = (res_inputs["hidden_states"].float() + res_inputs["residual"].float()).to(dtype)
    torch.testing.assert_close(res_out[1], summed, rtol=0, atol=0)
    row = summed[0].float()
    normed = row * torch.rsqrt(row.square().mean() + inputs["eps"]) * (1.0 + res_inputs["weight"].float())
    torch.testing.assert_close(res_out[0][0].float(), normed, rtol=2e-2, atol=2e-2)

    u.assert_unchanged(before, res_inputs)
    for output in res_out:
        u.assert_fresh(output, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
