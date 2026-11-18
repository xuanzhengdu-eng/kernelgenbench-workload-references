"""H800 accuracy/performance test for the SGLang CUDA branch of bf16 KV store.

Contract under test (SGLang 62384de1ae5e3bbd3159afd1639525b0bf39fbf9,
python/sglang/kernels/ops/kvcache/kvcache.py#L69-L120): K/V rows are flattened
``(tokens, heads*head_dim)`` and the cache rows are ``(slots, heads*head_dim)``.
The kernel skips writes to ``reserved_skip_index`` (0 is the CUDA-graph padding
slot); ``-1`` disables the skip.  The cache is mutated in place, sources are
read-only and the call returns None.
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
COMBOS = [(case, reserved) for case in CASES for reserved in (0, -1)]


def input_build(config):
    case, reserved = config
    tokens, blocks, block_size, heads, head_dim = case
    gen = torch.Generator(device="cpu").manual_seed(533)
    width = heads * head_dim
    kv = torch.randn((tokens, width), generator=gen, dtype=torch.bfloat16)
    value = -kv
    kv_cache = torch.randn(
        (blocks * block_size, width), generator=gen, dtype=torch.bfloat16
    )
    value_cache = -kv_cache
    return {
        "kv": kv,
        "kv_cache": kv_cache,
        "slot_mapping": torch.arange(tokens, dtype=torch.int64),
        "value": value,
        "value_cache": value_cache,
        "reserved_skip_index": reserved,
    }


def compute_ref(kv, value, kv_cache, value_cache, slot_mapping, reserved_skip_index=0):
    return kernelgenbench.baseline.v1_store_bf16_kv_cache_sgl_cuda(
        kv, value, kv_cache, value_cache, slot_mapping, reserved_skip_index
    )


def compute_res(kv, value, kv_cache, value_cache, slot_mapping, reserved_skip_index=0):
    return kernelgenbench.solution.v1_store_bf16_kv_cache_sgl_cuda(
        kv, value, kv_cache, value_cache, slot_mapping, reserved_skip_index
    )


def compare(ref_out, res_out, dtype=None):
    assert ref_out is None and res_out is None


def compute_ref_performance(**inputs):
    return bench_cuda(lambda: compute_ref(**inputs), warmup=5, rep=20)


def compute_res_performance(**inputs):
    return bench_cuda(lambda: compute_res(**inputs), warmup=5, rep=20)


@label("v1_store_bf16_kv_cache_sgl_cuda")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_store_bf16_kv_cache_sgl_cuda(config):
    case, reserved = config
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

    torch.testing.assert_close(
        res_inputs["kv_cache"], ref_inputs["kv_cache"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        res_inputs["value_cache"], ref_inputs["value_cache"], rtol=0, atol=0
    )
    for name, expected in read_only.items():
        assert torch.equal(ref_inputs[name], expected), name
        assert torch.equal(res_inputs[name], expected), name

    skipped = slots[slots == reserved].tolist()
    written = slots[slots != reserved].tolist()
    assert written
    for slot in written:
        assert torch.equal(res_inputs["kv_cache"][slot], res_inputs["kv"][slot])
        assert torch.equal(res_inputs["value_cache"][slot], res_inputs["value"][slot])
    for slot in skipped:
        assert torch.equal(ref_inputs["kv_cache"][slot], ref_before[0][slot])
        assert torch.equal(res_inputs["kv_cache"][slot], res_before[0][slot])

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
