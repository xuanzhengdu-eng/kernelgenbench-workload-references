"""Accuracy and performance for v1_vllm_moe_lora_align (vLLM CUDA op).

Official entry: ``vllm._custom_ops.moe_lora_align_block_size``, the small-batch
CUDA kernel used by the Punica LoRA wrapper.  It mutates three workspaces in
place and returns None.  The test checks per-adapter bucketing, block padding
with the ``numel`` sentinel, untouched inactive adapter slots, the expert map
(including -1 remaps) and that read-only inputs are unchanged.  Order inside an
expert bucket is not part of the contract, so buckets are compared as
multisets; the padded tail and the block table are compared exactly.
"""
import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# tokens, topk, experts, block, max_loras, adapter_enabled, lora_ids, expert_map
CASES = [
    (5, 2, 3, 4, 2, [1, 1], [-1, 0, 1], False),
    (9, 2, 4, 8, 3, [1, 0, 1], [-1, 0, 2, -1], True),
    (67, 2, 7, 16, 3, [1, 1, 1], [0, 1, 2, -1], False),
    (8, 1, 4, 2, 2, [0, 1], [-1, 0, 1], True),
    (4, 3, 2, 4, 2, [1, 1], [0, 1, -1], False),
]
COMBOS = [(case, index) for index, case in enumerate(CASES)]


def input_build(config):
    case, index = config
    tokens, topk, experts, block, max_loras, enabled, active, use_map = case
    device = "cpu"
    rng = torch.Generator(device=device).manual_seed(6970 + index)
    flat = torch.arange(tokens * topk, dtype=torch.int32)
    topk_ids = ((flat * 7 + flat // 3) % experts).reshape(tokens, topk)
    if tokens > 4:
        topk_ids.reshape(-1)[::11] = -1
    token_lora_mapping = torch.tensor(
        [(i % (max_loras + 1)) - 1 for i in range(tokens)], dtype=torch.int32
    )
    cap = tokens * topk + experts * (block - 1)
    blocks = (cap + block - 1) // block
    if use_map:
        expert_map = torch.tensor(
            [(-1 if i % 3 == 0 else (i + 1) % experts) for i in range(experts)],
            dtype=torch.int32,
        )
    else:
        expert_map = None
    return {
        "topk_ids": topk_ids.contiguous(),
        "token_lora_mapping": token_lora_mapping,
        "num_experts": experts,
        "block_size": block,
        "max_loras": max_loras,
        "max_num_tokens_padded": cap,
        "max_num_m_blocks": blocks,
        "sorted_token_ids": torch.full((max_loras * cap,), -777, dtype=torch.int32),
        "experts_ids": torch.full((max_loras * blocks,), -777, dtype=torch.int32),
        "num_tokens_post_pad": torch.full((max_loras,), -777, dtype=torch.int32),
        "adapter_enabled": torch.tensor(enabled, dtype=torch.int32),
        "lora_ids": torch.tensor(active, dtype=torch.int32),
        "expert_map": expert_map,
    }


def _to_device(inputs):
    out = {}
    for key, value in inputs.items():
        out[key] = value.to(kernelgenbench.device) if isinstance(value, torch.Tensor) else value
    return out


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_vllm_moe_lora_align(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_vllm_moe_lora_align(**kwargs)


def compare(ref_out, res_out, ref_inputs, res_inputs):
    assert ref_out is None and res_out is None
    cap = ref_inputs["max_num_tokens_padded"]
    blocks = ref_inputs["max_num_m_blocks"]
    block = ref_inputs["block_size"]
    count = ref_inputs["topk_ids"].numel()
    for slot in range(ref_inputs["max_loras"]):
        ref_total = int(ref_inputs["num_tokens_post_pad"][slot])
        res_total = int(res_inputs["num_tokens_post_pad"][slot])
        assert ref_total == res_total, f"slot {slot} total"
        ref_sorted = ref_inputs["sorted_token_ids"][slot * cap : (slot + 1) * cap]
        res_sorted = res_inputs["sorted_token_ids"][slot * cap : (slot + 1) * cap]
        ref_experts = ref_inputs["experts_ids"][slot * blocks : (slot + 1) * blocks]
        res_experts = res_inputs["experts_ids"][slot * blocks : (slot + 1) * blocks]
        if ref_total < 0:
            # Inactive/invalid adapter slot: untouched by both sides.
            assert torch.equal(ref_sorted, res_sorted)
            assert torch.equal(ref_experts, res_experts)
            continue
        assert torch.equal(ref_experts, res_experts)
        assert torch.equal(ref_sorted[ref_total:], res_sorted[ref_total:])
        assert bool((res_sorted[ref_total:] == count).all())
        for expert in range(ref_inputs["num_experts"]):
            positions = torch.nonzero(ref_experts == expert).reshape(-1)
            if positions.numel() == 0:
                continue
            start = int(positions[0]) * block
            end = (int(positions[-1]) + 1) * block
            bucket_ref = ref_sorted[start:end]
            bucket_res = res_sorted[start:end]
            assert torch.equal(bucket_ref.sort().values, bucket_res.sort().values)
            valid = bucket_res[bucket_res < count]
            assert bool((bucket_res[: len(valid)] < count).all())
            assert bool((bucket_res[len(valid) :] == count).all())
            assert torch.equal(valid.sort().values, bucket_ref[bucket_ref < count].sort().values)


def compute_ref_performance(**kwargs):
    return _bench(lambda: compute_ref(**kwargs))


def compute_res_performance(**kwargs):
    return _bench(lambda: compute_res(**kwargs))


def _bench(fn):
    import triton.testing

    return triton.testing.do_bench(fn, warmup=10, rep=50)


@label("v1_vllm_moe_lora_align")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_moe_lora_align(config):
    case, index = config
    tokens, topk, experts, block, max_loras, enabled, active, use_map = case
    inputs = input_build(config)
    assert inputs["topk_ids"].shape == (tokens, topk)
    assert inputs["topk_ids"].dtype == torch.int32 and inputs["topk_ids"].is_contiguous()
    assert inputs["max_num_tokens_padded"] == tokens * topk + experts * (block - 1)
    assert inputs["adapter_enabled"].shape == (max_loras,)
    assert inputs["lora_ids"].shape == (max_loras + 1,)
    assert inputs["expert_map"] is None or inputs["expert_map"].shape == (experts,)

    read_only = {
        key: value.clone()
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
        and key not in ("sorted_token_ids", "experts_ids", "num_tokens_post_pad")
    }
    ref_inputs = _to_device(inputs)
    res_inputs = _to_device(inputs)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs)
    for key, value in read_only.items():
        torch.testing.assert_close(res_inputs[key].cpu(), value, rtol=0, atol=0)
        torch.testing.assert_close(ref_inputs[key].cpu(), value, rtol=0, atol=0)

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
