"""H800 accuracy/performance test for the CUDA indexer-K store.

Contract under test (vLLM f358f0578a83f4e1fe6fc49a492d3ae9362de8e5,
vllm/_custom_ops.py#L3536-L3547, csrc/cache_kernels.cu#L2069-L2211 and the
inner kernel at csrc/cache_kernels.cu#L706-L780):
``indexer_k_quant_and_cache`` requires bf16 ``k`` with ``head_dim == 128`` and
stores row-major FP8 E4M3 values; the trailing 4 bytes per slot hold one fp32
scale, a UE8M0 exponent, or four packed MXFP8 exponent bytes (``mxfp8``).
Negative slots are skipped, only the cache is mutated and the call returns None.
The ROCm tiled-layout Triton variant has its own reference and test file.
"""
import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows532_541_test_utils import (
    bench_cuda,
    clone_tree,
)

# (blocks, block_size, dim, slots, input dtype, scale format)
COMBOS = [
    (2, 16, 128, (0, 15, 16, 31, -1), torch.bfloat16, "fp32"),
    (2, 16, 128, (17, 2, -1, 9), torch.bfloat16, "ue8m0"),
    (2, 16, 128, (0, 15, 16, 31, -1), torch.bfloat16, "mxfp8"),
    (8, 1, 128, (0, 3, -1, 7), torch.bfloat16, "fp32"),
    (2, 16, 128, (0, -1), torch.bfloat16, "mxfp8"),
]


def input_build(config):
    blocks, block_size, dim, slots, dtype, fmt = config
    count = len(slots)
    values = (
        (((torch.arange(count * dim, dtype=torch.float32) + 3) % 31) - 15).reshape(
            count, dim
        )
        / 8
    ).to(dtype)
    if slots == (0, -1):
        values.zero_()
    return {
        "k": values,
        "kv_cache": torch.full((blocks, block_size, dim + 4), 17, dtype=torch.uint8),
        "slot_mapping": torch.tensor(slots, dtype=torch.int64),
        "quant_block_size": 128,
        "scale_fmt": fmt,
    }


def compute_ref(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
    return kernelgenbench.baseline.v1_vllm_indexer_quant_store_cuda(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt
    )


def compute_res(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
    return kernelgenbench.solution.v1_vllm_indexer_quant_store_cuda(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt
    )


def compare(ref_out, res_out, dtype=None):
    assert ref_out is None and res_out is None


def value_slice(block_size, dim, offset):
    return slice(offset * dim, (offset + 1) * dim)


def scale_slice(block_size, dim, offset):
    return slice(block_size * dim + offset * 4, block_size * dim + (offset + 1) * 4)


def check_cache(ref_cache, res_cache, config):
    blocks, block_size, dim, slots, dtype, fmt = config
    ref_flat = ref_cache.view(blocks, -1)
    res_flat = res_cache.view(blocks, -1)
    touched = torch.zeros_like(ref_flat, dtype=torch.bool)
    for slot in slots:
        if slot < 0:
            continue
        page, offset = divmod(slot, block_size)
        touched[page, value_slice(block_size, dim, offset)] = True
        touched[page, scale_slice(block_size, dim, offset)] = True
        ref_bytes = ref_flat[page, value_slice(block_size, dim, offset)]
        res_bytes = res_flat[page, value_slice(block_size, dim, offset)]
        ref_scale_bytes = ref_flat[page, scale_slice(block_size, dim, offset)]
        res_scale_bytes = res_flat[page, scale_slice(block_size, dim, offset)]
        if fmt == "mxfp8":
            # The MXFP8 path stores exponent bytes only, so they must match.
            assert torch.equal(ref_scale_bytes, res_scale_bytes)
            # Four packed UE8M0 bytes, all equal to the shared scale exponent.
            assert bool((res_scale_bytes == res_scale_bytes[0]).all())
            ref_scale = torch.exp2(ref_scale_bytes[0].float() - 127)
            res_scale = torch.exp2(res_scale_bytes[0].float() - 127)
        else:
            ref_scale = ref_scale_bytes.view(torch.float32)[0]
            res_scale = res_scale_bytes.view(torch.float32)[0]
            torch.testing.assert_close(res_scale, ref_scale, rtol=2e-3, atol=1e-7)
        ref_values = (
            ref_bytes.contiguous().view(torch.float8_e4m3fn).float() * ref_scale
        )
        res_values = (
            res_bytes.contiguous().view(torch.float8_e4m3fn).float() * res_scale
        )
        torch.testing.assert_close(res_values, ref_values, rtol=3e-2, atol=2e-3)
    # Every byte outside the selected value/scale regions is untouched.
    assert torch.equal(ref_flat[~touched], res_flat[~touched])


def compute_ref_performance(**inputs):
    return bench_cuda(lambda: compute_ref(**inputs), warmup=5, rep=20)


def compute_res_performance(**inputs):
    return bench_cuda(lambda: compute_res(**inputs), warmup=5, rep=20)


@label("v1_vllm_indexer_quant_store_cuda")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_indexer_quant_store_cuda(config):
    blocks, block_size, dim, slots, dtype, fmt = config
    inputs = input_build(config)
    ref_inputs = clone_tree(inputs, kernelgenbench.device)
    res_inputs = clone_tree(inputs, kernelgenbench.device)
    ref_ptr = ref_inputs["kv_cache"].data_ptr()
    res_ptr = res_inputs["kv_cache"].data_ptr()
    ref_key = ref_inputs["k"].clone()
    res_key = res_inputs["k"].clone()
    ref_slots = ref_inputs["slot_mapping"].clone()
    res_slots = res_inputs["slot_mapping"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)
    if kernelgenbench.device.type == "cuda":
        torch.cuda.synchronize()
    assert ref_inputs["kv_cache"].data_ptr() == ref_ptr
    assert res_inputs["kv_cache"].data_ptr() == res_ptr
    check_cache(ref_inputs["kv_cache"], res_inputs["kv_cache"], config)
    assert torch.equal(ref_inputs["k"], ref_key) and torch.equal(res_inputs["k"], res_key)
    assert torch.equal(ref_inputs["slot_mapping"], ref_slots)
    assert torch.equal(res_inputs["slot_mapping"], res_slots)

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
