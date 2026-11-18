"""Accuracy and performance for v1_ds_softmax_context.

Operator: DeepSpeed ``softmax_context`` (QKV split, KV-cache write, masked and
windowed attention over the cache prefix) in both prompt and single-token
decode mode.
Official source (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/pt_binding.cpp#L439-L568

Contract: qkv [B,S,(H+2G)*D] FP32/FP16; cache buffers [B,H,capacity,D] and
output [B,S,H*D] same dtype/device/contiguous; G divides H; FP16 needs D%8==0.
Scores are scaled by ``norm_factor**2``; an optional additive [B,past+S] mask
and [B*H,past+S] ALiBi are added; triangular causality applies only when S>1;
an optional sliding window masks by real position.  Softmax runs in the qkv
dtype.  The prompt tokens are written to cache slots [past, past+S); later
slots are untouched and the previous prefix [0, past) is preserved.  The return
is ``(output_buffer, key_cache[:,:,:past+S], value_cache[:,:,:past+S])``, so the
first element aliases the caller's output buffer.  ``S>1`` requires past_len=0
(prompt); ``S==1`` appends at ``past_len`` (decode).  A first decode token with
past_len==0 has no prompt-written slot and is not reachable through the pinned
driver, so it is not part of the graded grid.

Scope: the ``layout='deepspeed'`` prompt and decode branches are graded.  The
``layout='sglang'`` address-table branch needs a real KV pool plus
req_to_token/req_pool_indices mapping and is recorded as unverified in the
report, not graded here.

Item: v1_ds_softmax_context (723-row 268).
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.upstream_timing import measure
from sandbox.verifier.test_parametrize import label, parametrize


# (B, S, H, G, D, capacity, causal, window_size, mask_mode, past_len)
# mask_mode: None keeps no mask/alibi; "mask" adds both; "half*" casts to fp16.
# past_len>0 means a single-token decode appended after that many prompt tokens,
# and the case also supplies the prompt tokens that seed the cache.
COMBOS = [
    (1, 4, 2, 2, 16, 8, True, 0, None, 0),
    (2, 8, 4, 2, 16, 12, True, 0, None, 0),
    (1, 3, 4, 1, 16, 7, False, 0, None, 0),
    (2, 6, 2, 2, 32, 10, False, 0, None, 0),
    (1, 6, 2, 2, 16, 8, True, 4, None, 0),
    (2, 4, 2, 2, 16, 8, True, 0, "mask", 0),
    (1, 4, 2, 2, 16, 8, True, 0, "half", 0),
    (1, 3, 4, 2, 16, 7, True, 0, "half_mask", 0),
    (2, 1, 4, 2, 16, 10, True, 0, None, 4),
    (1, 1, 2, 2, 16, 8, True, 0, None, 3),
    (2, 1, 4, 2, 16, 12, True, 0, "mask", 5),
    (1, 1, 4, 1, 16, 9, True, 0, "half", 4),
]

TOLERANCE = {torch.float32: (5e-6, 1e-6), torch.float16: (2e-3, 2e-3)}


def input_build(config):
    """Deterministic qkv batch, cache fills, optional mask/alibi and prompt.

    For past_len>0 the batch is a single decode token and ``past_qkv`` holds the
    prompt tokens that both sides push through the operator first, so the cache
    state being compared is produced by each side's own forward pass.  Caches
    start at -9 (K) and 11 (V), which also proves the unused tail stays intact.
    """
    B, S, H, G, D, capacity, causal, window, mode, past = config
    width = (H + 2 * G) * D

    def values(count, mul):
        flat = torch.arange(count, dtype=torch.float32)
        return ((((flat * mul) % 53) - 26) / 32).reshape(count)

    inputs = dict(
        qkv=values(B * S * width, 7).reshape(B, S, width),
        key_cache=torch.full((B, H, capacity, D), -9.0),
        value_cache=torch.full((B, H, capacity, D), 11.0),
        output_buffer=torch.full((B, S, H * D), 7.0),
        heads=H, kv_heads=G, norm_factor=D ** -0.25, triangular=causal)
    total = past + S
    if past:
        inputs["past_qkv"] = values(B * past * width, 11).reshape(B, past, width)
    if window:
        inputs["window_size"] = window
    if mode is not None and "mask" in mode:
        inputs["attn_mask"] = ((((torch.arange(B * total, dtype=torch.float32) * 5) % 11) - 5).reshape(B, total) / 7)
        inputs["alibi"] = ((((torch.arange(B * H * total, dtype=torch.float32) * 3) % 13) - 6).reshape(B * H, total) / 9)
    if mode is not None and mode.startswith("half"):
        for key in ("qkv", "past_qkv", "key_cache", "value_cache", "output_buffer",
                    "attn_mask", "alibi"):
            if key in inputs:
                inputs[key] = inputs[key].half()
    return inputs


def _run(entry, inputs):
    """One logical call: optional prompt prefill, then the graded tokens."""
    args = dict(inputs)
    prompt = args.pop("past_qkv", None)
    if prompt is not None:
        past_args = dict(args)
        past_args["qkv"] = prompt
        past_args["output_buffer"] = torch.full(
            (prompt.shape[0], prompt.shape[1], args["output_buffer"].shape[-1]),
            7.0, dtype=prompt.dtype, device=prompt.device)
        if "attn_mask" in past_args:
            past_args["attn_mask"] = args["attn_mask"][:, : prompt.shape[1]].contiguous()
            past_args["alibi"] = args["alibi"][:, : prompt.shape[1]].contiguous()
        entry(**past_args)
        args["past_len"] = prompt.shape[1]
    return entry(**args)


def compute_ref(**inputs):
    return _run(kernelgenbench.baseline.v1_ds_softmax_context, inputs)


def compute_res(**inputs):
    return _run(kernelgenbench.solution.v1_ds_softmax_context, inputs)


def compare(ref_out, res_out, config, ref_inputs, res_inputs):
    """Three aliased outputs with matching values; unused cache slots untouched."""
    B, S, H, G, D, capacity = config[:6]
    past = config[9]
    total = past + S
    dtype = res_inputs["qkv"].dtype
    assert isinstance(res_out, (tuple, list)) and len(res_out) == 3, \
        f"expected three outputs, got {type(res_out).__name__}"
    assert res_out[0] is res_inputs["output_buffer"], "output must alias output_buffer"
    for index, name in ((1, "key_cache"), (2, "value_cache")):
        assert torch._C._is_alias_of(res_out[index], res_inputs[name]), f"{name} must alias"
        assert res_out[index].shape == (B, H, total, D), (name, res_out[index].shape)
    rtol, atol = TOLERANCE[dtype] if dtype in TOLERANCE else (2e-3, 2e-3)
    for expected, actual in zip(ref_out, res_out):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.device == expected.device
        torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)
    # Untouched tail of every cache row keeps its fill value, in all modes.
    torch.testing.assert_close(res_inputs["key_cache"][:, :, total:],
                               torch.full_like(res_inputs["key_cache"][:, :, total:], -9), rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["value_cache"][:, :, total:],
                               torch.full_like(res_inputs["value_cache"][:, :, total:], 11), rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return measure(compute_ref, kwargs=inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, kwargs=inputs)


@label("v1_ds_softmax_context")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_softmax_context(config):
    inputs = input_build(config)
    device = kernelgenbench.device
    B, S, H, G, D, capacity = config[:6]
    past = config[9]
    assert inputs["qkv"].shape == (B, S, (H + 2 * G) * D)
    assert inputs["key_cache"].shape == (B, H, capacity, D)
    assert capacity >= past + S
    if past:
        assert inputs["past_qkv"].shape == (B, past, (H + 2 * G) * D)

    def move(values):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in values.items()}

    ref_inputs, res_inputs = move(inputs), move(inputs)
    read_only = {k: v.clone() for k, v in res_inputs.items()
                 if isinstance(v, torch.Tensor) and k in ("qkv", "past_qkv", "attn_mask", "alibi")}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, ref_inputs, res_inputs)
    for key, value in read_only.items():
        torch.testing.assert_close(res_inputs[key], value, rtol=0, atol=0)
    # Non-finite scores only where a whole row is masked and S>1; decode rows
    # always keep at least one visible slot.
    assert torch.isfinite(res_out[0]).all(), "unexpected non-finite output"

    device_inputs = move(inputs)
    ref_time = compute_ref_performance(**device_inputs)
    device_inputs = move(inputs)
    res_time = compute_res_performance(**device_inputs)
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult

    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
