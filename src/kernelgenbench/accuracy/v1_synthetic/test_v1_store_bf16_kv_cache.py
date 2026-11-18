"""H800 accuracy/performance test for the vLLM branch of v1_store_bf16_kv_cache.

Contract under test (vLLM f358f0578a83f4e1fe6fc49a492d3ae9362de8e5,
vllm/v1/attention/ops/triton_reshape_and_cache_flash.py#L377-L484):
kv/value ``(tokens, heads, head_dim)`` BF16 and caches
``(blocks, block_size, heads, head_dim)``; int64 slot ids, where -1 is a padded
token that must not be written.  The cache is mutated in place and the call
returns None.  SGLang CPU/CUDA and Megatron have different signatures/shapes and
their own reference and test files.
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

# (tokens, blocks, block_size, heads, head_dim)
CASES = [
    (16, 4, 16, 1, 128),
    (32, 8, 16, 2, 128),
    (8, 2, 32, 1, 64),
]
COMBOS = [(case, slot0_valid) for case in CASES for slot0_valid in (True, False)]


def input_build(config):
    case, slot0_valid = config
    tokens, blocks, block_size, heads, head_dim = case
    gen = torch.Generator(device="cpu").manual_seed(533)
    kv = torch.randn((tokens, heads, head_dim), generator=gen, dtype=torch.bfloat16)
    value = -kv
    kv_cache = torch.randn(
        (blocks, block_size, heads, head_dim), generator=gen, dtype=torch.bfloat16
    )
    value_cache = -kv_cache
    slots = torch.arange(tokens, dtype=torch.int64)
    slots[1::7] = -1
    if not slot0_valid:
        slots[0] = -1
    return {
        "kv": kv,
        "kv_cache": kv_cache,
        "slot_mapping": slots,
        "value": value,
        "value_cache": value_cache,
    }


def compute_ref(kv, kv_cache, slot_mapping, value, value_cache):
    return kernelgenbench.baseline.v1_store_bf16_kv_cache(
        kv, kv_cache, slot_mapping, value, value_cache
    )


def compute_res(kv, kv_cache, slot_mapping, value, value_cache):
    return kernelgenbench.solution.v1_store_bf16_kv_cache(
        kv, kv_cache, slot_mapping, value, value_cache
    )


def compare(ref_out, res_out, dtype=None):
    assert ref_out is None and res_out is None


def compute_ref_performance(**inputs):
    return bench_cuda(lambda: compute_ref(**inputs), warmup=5, rep=20)


def compute_res_performance(**inputs):
    return bench_cuda(lambda: compute_res(**inputs), warmup=5, rep=20)


@label("v1_store_bf16_kv_cache")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_store_bf16_kv_cache(config):
    case, slot0_valid = config
    tokens, blocks, block_size, heads, head_dim = case
    inputs = input_build(config)
    ref_inputs = clone_tree(inputs, kernelgenbench.device)
    res_inputs = clone_tree(inputs, kernelgenbench.device)
    ref_ptrs = (ref_inputs["kv_cache"].data_ptr(), ref_inputs["value_cache"].data_ptr())
    res_ptrs = (res_inputs["kv_cache"].data_ptr(), res_inputs["value_cache"].data_ptr())
    ref_before = (ref_inputs["kv_cache"].clone(), ref_inputs["value_cache"].clone())
    res_before = (res_inputs["kv_cache"].clone(), res_inputs["value_cache"].clone())
    slots = inputs["slot_mapping"]
    read_only = {
        name: clone_tree(inputs[name], kernelgenbench.device)
        for name in ("kv", "value", "slot_mapping")
    }

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)
    assert ref_inputs["kv_cache"].data_ptr() == ref_ptrs[0]
    assert res_inputs["kv_cache"].data_ptr() == res_ptrs[0]
    assert ref_inputs["value_cache"].data_ptr() == ref_ptrs[1]
    assert res_inputs["value_cache"].data_ptr() == res_ptrs[1]

    # Both sides write identical K and V destinations, byte for byte.
    torch.testing.assert_close(
        res_inputs["kv_cache"], ref_inputs["kv_cache"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        res_inputs["value_cache"], ref_inputs["value_cache"], rtol=0, atol=0
    )
    for name, expected in read_only.items():
        assert torch.equal(ref_inputs[name], expected), name
        assert torch.equal(res_inputs[name], expected), name

    # Every negative slot is padding: its cache row is left untouched.
    invalid = slots[slots < 0].tolist()
    assert invalid
    for slot in invalid:
        block, offset = divmod(slot, block_size)
        assert torch.equal(
            ref_inputs["kv_cache"][block, offset], ref_before[0][block, offset]
        )
        assert torch.equal(
            res_inputs["kv_cache"][block, offset], res_before[0][block, offset]
        )

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
