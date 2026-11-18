"""Accuracy and performance for v1_vllm_gptq_marlin_repack (row 374).

Official sources:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L1110-L1116
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L1165-L1182

The GPTQ input is int32 ``[size_k / pack, size_n]`` (K-major nibbles), or
``[experts, size_k / pack, size_n]`` for a batched expert weight, which upstream
repacks with ``gptq_marlin_moe_repack`` by running the same 2-D CUDA kernel per
expert and stacking. Output is int32 ``[size_k / 16, size_n * 16 / pack]`` (or
``[experts, ...]``) in the Marlin tile order, freshly allocated and contiguous;
the input is unchanged.

Each COMBOS entry is ``(experts, size_k, size_n, num_bits)``; ``experts == 0``
means a 2-D weight.
"""
import statistics
import time

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

NAME = "v1_vllm_gptq_marlin_repack"

COMBOS = [
    (0, 16, 64, 4),
    (0, 32, 128, 4),
    (0, 48, 192, 4),
    (0, 32, 128, 8),
    (0, 16, 64, 8),
    (2, 64, 128, 4),
    (3, 32, 64, 8),
]


def input_build(config):
    experts, size_k, size_n, num_bits = config
    pack = 32 // num_bits
    shape = (size_k // pack, size_n)
    generator = torch.Generator().manual_seed(3700 + size_k + size_n + num_bits)
    words = torch.randint(0, 2**31 - 1, shape, generator=generator, dtype=torch.int64)
    if experts:
        words = words.unsqueeze(0).repeat(experts, 1, 1)
        # Distinct content per expert so a wrong expert order cannot pass.
        words = (words + torch.arange(experts, dtype=torch.int64).view(-1, 1, 1)) & 0xFFFFFFFF
    return {
        "b_q_weight": words.to(torch.int32).contiguous(),
        "size_k": size_k,
        "size_n": size_n,
        "num_bits": num_bits,
    }


def _fresh(inputs, device):
    out = {name: value for name, value in inputs.items() if not torch.is_tensor(value)}
    out["b_q_weight"] = inputs["b_q_weight"].to(device=device, copy=True)
    return out


def compute_ref(b_q_weight, size_k, size_n, num_bits):
    return kernelgenbench.baseline.v1_vllm_gptq_marlin_repack(b_q_weight, size_k, size_n, num_bits)


def compute_res(b_q_weight, size_k, size_n, num_bits):
    return kernelgenbench.solution.v1_vllm_gptq_marlin_repack(b_q_weight, size_k, size_n, num_bits)


def _value_histogram(tensor, num_bits):
    """Count every packed value; a correct repack preserves this multiset."""
    pack = 32 // num_bits
    words = tensor.to(torch.int64) & 0xFFFFFFFF
    values = torch.stack(
        [(words >> (i * num_bits)) & ((1 << num_bits) - 1) for i in range(pack)], -1
    ).reshape(-1)
    return torch.bincount(values, minlength=1 << num_bits)


def _measure(fn, config):
    device = kernelgenbench.device
    samples = []
    for _ in range(8):
        arguments = _fresh(input_build(config), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn(**arguments)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples[1:])


@label(NAME)
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_gptq_marlin_repack(config):
    experts, size_k, size_n, num_bits = config
    pack = 32 // num_bits
    inputs = input_build(config)
    ref_inputs = _fresh(inputs, kernelgenbench.device)
    res_inputs = _fresh(inputs, kernelgenbench.device)
    before = res_inputs["b_q_weight"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)

    expected_shape = (size_k // 16, size_n * 16 // pack)
    if experts:
        expected_shape = (experts,) + expected_shape
    for out in (ref_out, res_out):
        assert isinstance(out, torch.Tensor)
        assert out.shape == expected_shape and out.dtype == torch.int32
        assert out.is_contiguous()
        assert not torch._C._is_alias_of(out, res_inputs["b_q_weight"])
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)
    # A correct repack permutes the quantized values: the multiset must survive.
    torch.testing.assert_close(
        _value_histogram(res_out, num_bits),
        _value_histogram(res_inputs["b_q_weight"], num_bits),
        rtol=0,
        atol=0,
    )

    torch.testing.assert_close(before, res_inputs["b_q_weight"], rtol=0, atol=0)
    torch.testing.assert_close(ref_inputs["b_q_weight"].cpu(), inputs["b_q_weight"], rtol=0, atol=0)

    ref_time = _measure(compute_ref, config)
    res_time = _measure(compute_res, config)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
