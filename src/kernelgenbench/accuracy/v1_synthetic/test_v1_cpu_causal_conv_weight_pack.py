"""Bit-exact CPU causal-conv packed-weight test (vLLM entry).

Item: v1_cpu_causal_conv_weight_pack (723-row 413), vLLM reference.
Official source (pinned commit f358f057):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/cpu/sgl-kernels/conv.cpp#L543-L585

Input: contiguous CPU fp16/bf16 ``weight[N, 4]`` with ``N % 32 == 0`` (0 is
legal).  The payload is opaque 16-bit data, so the assertion compares int16
views bit-for-bit.  The official x86 operator has no ``block_n`` argument; the
ARM ``BLOCK_N=8`` build variant is not present in the pinned wheel and is
recorded as an unverified backend in the report.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows412_421_checks as checks
from sandbox.verifier.test_parametrize import label, parametrize


# N, dtype, storage offset in 16-bit words.
CASES = [
    (32, torch.bfloat16, 0),
    (64, torch.float16, 2),
    (96, torch.bfloat16, 2),
    (0, torch.float16, 0),
    (128, torch.bfloat16, 4),
]
COMBOS = CASES


def input_build(case):
    n, dtype, offset = case
    raw = ((torch.arange(n * 4 + offset, dtype=torch.int32) * 15917 + 0x7F81) % 65536).to(torch.int16)
    if n:
        patterns = torch.tensor(
            [0x7F81, 0x7FC1, 0x7F80, 0xFF81, 0x8000, 0x7FFF, 0, 0xFFFF],
            dtype=torch.int32,
        ).to(torch.int16)
        raw[offset : offset + len(patterns)] = patterns
    weight = raw[offset:].view(dtype).reshape(n, 4)
    return {"weight": weight}


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_cpu_causal_conv_weight_pack(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_cpu_causal_conv_weight_pack(**kwargs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert ref_out.shape == res_out.shape and ref_out.dtype == res_out.dtype == dtype
    assert res_out.device.type == "cpu" and res_out.is_contiguous()
    assert torch.equal(ref_out.view(torch.int16), res_out.view(torch.int16))


@label("v1_cpu_causal_conv_weight_pack")
@parametrize("case", COMBOS)
@pytest.mark.parametrize("case", COMBOS)
def test_accuracy_v1_cpu_causal_conv_weight_pack(case):
    n, dtype, _ = case
    inputs = input_build(case)
    ref_inputs = {k: v.clone() for k, v in inputs.items()}
    res_inputs = {k: v.clone() for k, v in inputs.items()}
    ref_before = ref_inputs["weight"].view(torch.int16).clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    assert n == case[0]
    assert torch.equal(ref_before, ref_inputs["weight"].view(torch.int16))
    assert torch.equal(ref_before, res_inputs["weight"].view(torch.int16))
    assert not torch._C._is_alias_of(res_out, res_inputs["weight"])

    ref_time = checks.measure(
        compute_ref, lambda: input_build(case), torch.device("cpu")
    )
    res_time = checks.measure(
        compute_res, lambda: input_build(case), torch.device("cpu")
    )
    return checks.timings(ref_time, res_time)
