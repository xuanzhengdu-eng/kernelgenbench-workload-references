"""Accuracy and performance for v1_vllm_deepgemm_moe_permute.

vLLM DeepGEMM activation permutation contract:

* ``aq`` is ``[tokens, hidden]`` FP8, ``aq_scale`` is ``[tokens, hidden //
  block_size]`` float32 or uint8 (UE8M0), ``topk_ids`` is signed int32/int64;
  ``-1`` marks an invalid route and ``expert_map`` may remap global ids to
  local ids (``-1`` drops the route);
* each local expert's rows are grouped and padded to 128 rows; the returned
  layout is either one expert id per padded row (``-1`` padding) or, with
  ``use_psum_layout``, one end offset per expert;
* ``inv`` maps each valid ``(token, topk)`` route to its row; the order inside
  an expert is unspecified, so the test compares through that mapping;
* uint8 scales are packed four per int32 into a TMA-aligned MN-major layout,
  float32 scales are copied row-major.

The Solution is the pinned vLLM entry
``vllm.model_executor.layers.fused_moe.deep_gemm_utils.deepgemm_moe_permute``
with an ``ExpertTokensMetadata`` carrying the exact per-expert counts, so the
allocation matches the reference's exact-count contract; the real Triton
``_fwd_kernel_ep_scatter_1/2`` kernels perform grouping, packing and inverse
index writes.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows562_571_utils import fresh, measure, result
from sandbox.verifier.test_parametrize import label, parametrize


# (tokens, hidden, topk, local experts, expert map, uint8 scales, psum layout)
COMBOS = [
    (5, 128, 2, 3, False, False, False),
    (17, 256, 3, 4, True, False, True),
    (131, 128, 2, 3, False, False, True),
    (9, 160, 3, 3, True, True, False),
    (19, 256, 2, 5, False, True, True),
    (2, 32, 1, 4, True, True, False),
]


def input_build(config):
    tokens, hidden, topk, experts, use_map, packed, _ = config
    generator = torch.Generator(device="cpu").manual_seed(531 + tokens + hidden + topk)
    aq = torch.randn((tokens, hidden), generator=generator).to(torch.float8_e4m3fn)
    block = 32 if packed else 128
    if packed:
        scale = torch.randint(0, 256, (tokens, hidden // block), generator=generator, dtype=torch.uint8)
    else:
        scale = torch.rand((tokens, hidden // block), generator=generator, dtype=torch.float32)
    global_experts = experts + 2 if use_map else experts
    ids = torch.randint(0, global_experts, (tokens, topk), generator=generator, dtype=torch.int32)
    if ids.numel() > 3:
        ids.reshape(-1)[3::7] = -1
    expert_map = (
        torch.tensor([i if i < experts else -1 for i in range(global_experts)], dtype=torch.int32)
        if use_map
        else None
    )
    return {
        "aq": aq,
        "aq_scale": scale,
        "topk_ids": ids,
        "local_num_experts": experts,
        "expert_map": expert_map,
        "block_size": block,
        "use_psum_layout": config[6],
    }


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_vllm_deepgemm_moe_permute(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_vllm_deepgemm_moe_permute(**kwargs)


def group_counts(inputs):
    mapping = None if inputs["expert_map"] is None else inputs["expert_map"].cpu().tolist()
    groups = [[] for _ in range(inputs["local_num_experts"])]
    for token, row in enumerate(inputs["topk_ids"].cpu().tolist()):
        for slot, expert in enumerate(row):
            if expert == -1:
                continue
            if mapping is not None:
                expert = mapping[expert]
            if expert >= 0:
                groups[expert].append((token, slot))
    return groups


def compare(ref_out, res_out, inputs):
    aq, ids = inputs["aq"], inputs["topk_ids"]
    ref_x, ref_scale, ref_layout, ref_inv, ref_align = ref_out
    res_x, res_scale, res_layout, res_inv, res_align = res_out

    assert ref_align == res_align == 128
    assert res_x.shape == ref_x.shape and res_x.shape[1] == aq.shape[1]
    assert res_x.dtype == ref_x.dtype == aq.dtype
    assert res_x.device == aq.device
    assert res_scale.dtype == ref_scale.dtype
    assert res_scale.shape == ref_scale.shape
    assert res_scale.stride() == ref_scale.stride()
    assert res_layout.dtype == ref_layout.dtype == torch.int32
    assert res_layout.shape == ref_layout.shape
    assert res_inv.dtype == ref_inv.dtype == torch.int32
    assert res_inv.shape == ref_inv.shape == ids.shape
    torch.testing.assert_close(res_layout, ref_layout, rtol=0, atol=0)

    groups = group_counts(inputs)
    if inputs["use_psum_layout"]:
        # One entry per expert: unpadded end offset of that expert's rows.
        expected = []
        cursor = 0
        for group in groups:
            expected.append(cursor + len(group))
            cursor += ((len(group) + 127) // 128) * 128
        assert res_layout.cpu().tolist() == expected
    start = 0
    for expert, group in enumerate(groups):
        count = len(group)
        rows = set()
        for token, slot in group:
            ref_pos = int(ref_inv[token, slot])
            res_pos = int(res_inv[token, slot])
            assert start <= res_pos < start + count
            assert start <= ref_pos < start + count
            rows.add(res_pos)
            torch.testing.assert_close(
                res_x[res_pos].view(torch.uint8), aq[token].view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(
                ref_x[ref_pos].view(torch.uint8), aq[token].view(torch.uint8), rtol=0, atol=0
            )
            if inputs["aq_scale"].dtype == torch.uint8:
                raw = inputs["aq_scale"][token].cpu().tolist()
                packed = res_scale[res_pos].cpu().tolist()
                for index, value in enumerate(raw):
                    assert ((packed[index // 4] & 0xFFFFFFFF) >> (8 * (index % 4))) & 255 == value
                ref_packed = ref_scale[ref_pos].cpu().tolist()
                assert ref_packed == packed
            else:
                torch.testing.assert_close(res_scale[res_pos], inputs["aq_scale"][token], rtol=0, atol=0)
                torch.testing.assert_close(ref_scale[ref_pos], inputs["aq_scale"][token], rtol=0, atol=0)
        assert rows == set(range(start, start + count))
        if not inputs["use_psum_layout"]:
            padded = ((count + 127) // 128) * 128
            if padded > count:
                assert bool((res_layout[start + count:start + padded] == -1).all())
        start += ((count + 127) // 128) * 128


def compute_ref_performance(**kwargs):
    return measure(compute_ref, kwargs, kernelgenbench.device, warmup=3, reps=10)


def compute_res_performance(**kwargs):
    return measure(compute_res, kwargs, kernelgenbench.device, warmup=3, reps=10)


@label("v1_vllm_deepgemm_moe_permute")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_deepgemm_moe_permute(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs)
    # Inputs are read-only: aq/aq_scale/topk_ids must be unchanged bit for bit.
    for name in ("aq", "aq_scale", "topk_ids"):
        if res_inputs[name].dtype == torch.float8_e4m3fn:
            torch.testing.assert_close(
                res_inputs[name].view(torch.uint8).cpu(),
                inputs[name].view(torch.uint8),
                rtol=0,
                atol=0,
            )
        else:
            torch.testing.assert_close(res_inputs[name].cpu(), inputs[name], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return result(ref_time, res_time)
