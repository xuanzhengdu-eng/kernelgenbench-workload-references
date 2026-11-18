"""Accuracy and performance for v1_merge_attn_states (vLLM Triton merge).

Shapes: output/prefix_output/suffix_output [tokens, heads, head_dim] in
FP32/FP16/BF16; prefix_lse/suffix_lse [heads, tokens] in FP32; output_lse
[heads, tokens] FP32 is optional.  ``prefill_tokens_with_context`` selects all,
half, or none of the tokens.  Both output buffers are caller-owned and mutated
in place; the return value is None.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.activation_rows11_21_checks import (
    fresh, measure,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


COMBOS = [
    (shape, dtype, context, with_lse)
    for shape in ((7, 2, 33), (16, 4, 64))
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for context in ("all", "partial", "none")
    for with_lse in (False, True)
]


def input_build(config):
    (tokens, heads, head_dim), dtype, context, with_lse = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    prefix = torch.randn((tokens, heads, head_dim), generator=generator, dtype=dtype)
    suffix = torch.randn((tokens, heads, head_dim), generator=generator, dtype=dtype)
    plse = torch.randn((heads, tokens), generator=generator)
    slse = torch.randn((heads, tokens), generator=generator)
    plse[0, 0] = float("-inf")
    slse[0, 0] = float("-inf")
    plse[1, 1] = float("inf")
    return {
        "output": torch.randn((tokens, heads, head_dim), generator=generator, dtype=dtype),
        "prefix_output": prefix,
        "prefix_lse": plse,
        "suffix_output": suffix,
        "suffix_lse": slse,
        "output_lse": torch.randn((heads, tokens), generator=generator) if with_lse else None,
        "prefill_tokens_with_context": {"all": None, "partial": tokens // 2, "none": 0}[context],
    }


def compute_ref(output, prefix_output, prefix_lse, suffix_output, suffix_lse,
                output_lse=None, prefill_tokens_with_context=None):
    return kernelgenbench.baseline.v1_merge_attn_states(
        output, prefix_output, prefix_lse, suffix_output, suffix_lse,
        output_lse=output_lse,
        prefill_tokens_with_context=prefill_tokens_with_context)


def compute_res(output, prefix_output, prefix_lse, suffix_output, suffix_lse,
                output_lse=None, prefill_tokens_with_context=None):
    return kernelgenbench.solution.v1_merge_attn_states(
        output, prefix_output, prefix_lse, suffix_output, suffix_lse,
        output_lse=output_lse,
        prefill_tokens_with_context=prefill_tokens_with_context)


def compare(ref_out, res_out, config, ref_inputs, res_inputs, pointers):
    _shape, dtype, _context, with_lse = config
    assert ref_out is None and res_out is None
    assert res_inputs["output"].data_ptr() == pointers["output"]
    if with_lse:
        assert res_inputs["output_lse"].data_ptr() == pointers["output_lse"]
    rtol, atol = {
        torch.float32: (2e-5, 2e-5),
        torch.float16: (4e-3, 4e-3),
        torch.bfloat16: (3e-2, 3e-2),
    }[dtype]
    torch.testing.assert_close(res_inputs["output"], ref_inputs["output"],
                               rtol=rtol, atol=atol)
    if with_lse:
        assert res_inputs["output_lse"].dtype == torch.float32
        torch.testing.assert_close(res_inputs["output_lse"], ref_inputs["output_lse"],
                                   rtol=2e-5, atol=2e-5)
    for name in ("prefix_output", "prefix_lse", "suffix_output", "suffix_lse"):
        torch.testing.assert_close(res_inputs[name], ref_inputs[name], atol=0, rtol=0)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_merge_attn_states")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_merge_attn_states(config):
    inputs = input_build(config)
    device_inputs = fresh(inputs, kernelgenbench.device)
    ref_inputs = fresh(device_inputs, kernelgenbench.device)
    res_inputs = fresh(device_inputs, kernelgenbench.device)
    pointers = {"output": res_inputs["output"].data_ptr()}
    if config[3]:
        pointers["output_lse"] = res_inputs["output_lse"].data_ptr()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, ref_inputs, res_inputs, pointers)

    # The prefix/suffix inputs are read-only; repeated writes to output are
    # idempotent, so the original snapshot can be reused for timing.
    for name in ("prefix_output", "prefix_lse", "suffix_output", "suffix_lse"):
        torch.testing.assert_close(res_inputs[name], device_inputs[name], atol=0, rtol=0)

    ref_time = compute_ref_performance(**device_inputs)
    res_time = compute_res_performance(**device_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
