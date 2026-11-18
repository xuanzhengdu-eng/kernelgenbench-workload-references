"""H800 test for the Megatron torch fallback of v1_store_bf16_kv_cache.

Contract under test (NVIDIA Megatron-LM
266f1c97dca477ca8b92c16087506da2000b0b84,
megatron/core/inference/contexts/dynamic_context.py#L963-L1011): when the fused
Triton helper is unavailable, ``DynamicInferenceContext.append_key_value_cache``
squeezes the singleton sequence axis and assigns the first
``padded_active_token_count`` rows with advanced indexing:
``memory_buffer[0, layer, block_idx, local_pos] = key[:n]`` and likewise for the
value plane.  ``layer_number`` is mapped through ``layer_map`` first, and the
buffer is mutated in place with no return value.
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

# (tokens, blocks, block_size, heads, head dim, active rows)
COMBOS = [
    (6, 4, 16, 2, 8, "full"),
    (8, 4, 16, 1, 16, "partial"),
    (5, 3, 8, 2, 4, "zero"),
]


def input_build(config):
    tokens, blocks, block_size, heads, head_dim, mode = config
    gen = torch.Generator(device="cpu").manual_seed(5331)
    kv = torch.randn((tokens, 1, heads, head_dim), generator=gen, dtype=torch.bfloat16)
    value = torch.randn(
        (tokens, 1, heads, head_dim), generator=gen, dtype=torch.bfloat16
    )
    memory = torch.randn(
        (2, 3, blocks, block_size, heads, head_dim),
        generator=gen,
        dtype=torch.bfloat16,
    )
    slots = torch.randperm(blocks * block_size, generator=gen)[:tokens]
    active = {"full": tokens, "partial": max(1, tokens // 2), "zero": 0}[mode]
    return {
        "kv": kv,
        "value": value,
        "memory_buffer": memory,
        "layer_number": 1,
        "padded_active_token_count": active,
        "token_to_block_idx": slots // block_size,
        "token_to_local_position_within_kv_block": slots % block_size,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_store_bf16_kv_cache_megatron_torch(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_store_bf16_kv_cache_megatron_torch(**inputs)


def compare(ref_out, res_out, dtype=None):
    assert ref_out is None and res_out is None


def compute_ref_performance(**inputs):
    return bench_cuda(lambda: compute_ref(**inputs), warmup=5, rep=20)


def compute_res_performance(**inputs):
    return bench_cuda(lambda: compute_res(**inputs), warmup=5, rep=20)


@label("v1_store_bf16_kv_cache_megatron_torch")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_store_bf16_kv_cache_megatron_torch(config):
    inputs = input_build(config)
    active = inputs["padded_active_token_count"]
    ref_inputs = clone_tree(inputs, kernelgenbench.device)
    res_inputs = clone_tree(inputs, kernelgenbench.device)
    ref_ptr = ref_inputs["memory_buffer"].data_ptr()
    res_ptr = res_inputs["memory_buffer"].data_ptr()
    ref_before = ref_inputs["memory_buffer"].clone()
    res_before = res_inputs["memory_buffer"].clone()
    read_only = {
        name: clone_tree(inputs[name], kernelgenbench.device)
        for name in ("kv", "value", "token_to_block_idx", "token_to_local_position_within_kv_block")
    }

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)
    assert ref_inputs["memory_buffer"].data_ptr() == ref_ptr
    assert res_inputs["memory_buffer"].data_ptr() == res_ptr
    torch.testing.assert_close(
        res_inputs["memory_buffer"], ref_inputs["memory_buffer"], rtol=0, atol=0
    )
    if active == 0:
        assert torch.equal(res_inputs["memory_buffer"], res_before)
        assert torch.equal(ref_inputs["memory_buffer"], ref_before)
    else:
        blocks = res_inputs["token_to_block_idx"][:active].long()
        local = res_inputs["token_to_local_position_within_kv_block"][:active].long()
        for layer in (0, 1):
            source = res_inputs["kv"] if layer == 0 else res_inputs["value"]
            assert torch.equal(
                res_inputs["memory_buffer"][layer, 1, blocks, local], source[:active, 0]
            )
        # Inactive padded token rows keep their original cache values.
        if active < inputs["kv"].shape[0]:
            rest = res_inputs["token_to_block_idx"][active:].long()
            rest_local = res_inputs["token_to_local_position_within_kv_block"][active:].long()
            for layer in (0, 1):
                assert torch.equal(
                    res_inputs["memory_buffer"][layer, 1, rest, rest_local],
                    res_before[layer, 1, rest, rest_local],
                )
        for name, expected in read_only.items():
            assert torch.equal(ref_inputs[name], expected), name
            assert torch.equal(res_inputs[name], expected), name

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
