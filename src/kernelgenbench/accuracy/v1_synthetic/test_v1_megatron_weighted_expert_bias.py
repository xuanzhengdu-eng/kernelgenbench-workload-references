"""Accuracy and performance for v1_megatron_weighted_expert_bias.

Contract: ``TEGroupedMLP._apply_bias``. Rows are split by ``tokens_per_expert``
(a Python list, as in the caller), each chunk gets its expert bias row scaled by
that expert's probability slice, then the chunks are concatenated and the input
shape restored. ``bias_parallel is None`` returns the identical input object.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import moe_rows222_231_checks as checks


# (expert token counts, width, input dtype, probability dtype, bias present,
#  leading 3-D input)
COMBOS = [
    ((2, 0, 3), 8, torch.float32, torch.float32, False, False),
    ((1, 2, 1), 7, torch.bfloat16, torch.float32, False, True),
    ((0, 3, 0), 16, torch.bfloat16, torch.bfloat16, False, False),
    ((2, 1, 1), 4, torch.float32, torch.float32, True, True),
    ((0, 0, 0), 5, torch.float16, torch.float32, False, False),
    ((3, 2, 1, 0), 6, torch.float16, torch.float16, False, False),
]


def input_build(config):
    counts, width, dtype, prob_dtype, has_bias, three_d = config
    generator = torch.Generator(device="cpu").manual_seed(224)
    total = sum(counts)
    shape = (2, total // 2, width) if three_d else (total, width)
    intermediate = torch.randn(shape, dtype=dtype, generator=generator)
    bias = (
        torch.randn((len(counts), width), dtype=dtype, generator=generator)
        if has_bias
        else None
    )
    permuted_probs = torch.rand((total, 1), dtype=prob_dtype, generator=generator)
    return {
        "intermediate_parallel": intermediate,
        "bias_parallel": bias,
        "tokens_per_expert": list(counts),
        "permuted_probs": permuted_probs,
    }


def compute_ref(intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs):
    return kernelgenbench.baseline.v1_megatron_weighted_expert_bias(
        intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs
    )


def compute_res(intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs):
    return kernelgenbench.solution.v1_megatron_weighted_expert_bias(
        intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs
    )


def compare(ref_out, res_out, dtype=None, bias_present=False):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape and res_out.dtype == ref_out.dtype == dtype
    rtol, atol = checks.tolerance(dtype)
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_megatron_weighted_expert_bias, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_megatron_weighted_expert_bias, inputs, device=kernelgenbench.device)


@label("v1_megatron_weighted_expert_bias")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_weighted_expert_bias(config):
    counts, width, dtype, prob_dtype, has_bias, three_d = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    before = {
        name: value.cpu().clone()
        for name, value in res_inputs.items()
        if isinstance(value, torch.Tensor)
    }

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype, has_bias)

    # No bias is the identity branch in the source: the same object comes back.
    if not has_bias:
        assert res_out is res_inputs["intermediate_parallel"]
        assert ref_out is ref_inputs["intermediate_parallel"]
    else:
        # Every expert chunk is offset by its own probability-scaled bias row.
        view = res_inputs["intermediate_parallel"].reshape(-1, width)
        offset = 0
        for expert, count in enumerate(counts):
            if count:
                expected = (
                    view[offset : offset + count]
                    + res_inputs["bias_parallel"][expert]
                    * res_inputs["permuted_probs"][offset : offset + count]
                ).to(dtype)
                torch.testing.assert_close(
                    res_out.reshape(-1, width)[offset : offset + count],
                    expected,
                    rtol=checks.tolerance(dtype)[0],
                    atol=checks.tolerance(dtype)[1],
                )
            offset += count
    for name, value in before.items():
        assert torch.equal(res_inputs[name].cpu(), value), f"{name} was modified"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
