"""Accuracy and performance for v1_vllm_indexer_q_rope_quant.

Both sides receive identical (positions, q, cos_sin_cache, weights,
softmax_scale, head_scale, is_neox).  q is BF16 [tokens, heads, 128].  Lanes
0:64 are rotated with the 32 cos lanes and 32 sin lanes of the cache row named
by ``positions``; NeoX layout rotates (0:32, 32:64) pairs, the interleaved
layout rotates (even, odd) pairs.  The rotated halves are rounded back to BF16
before the amax over all 128 lanes selects the e8m0 scale
``2**ceil(log2(max(amax, 1e-10) / 448))``.  Lanes 64:128 are copied unscaled.
Outputs: FP8_E4M3 [tokens, heads, 128] and FP32 weights multiplied by
``q_scale * softmax_scale * head_scale``.
"""
import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# tokens, heads, rotary layout, q layout, value pattern
COMBOS = [
    (3, 2, True, "contiguous", "ramp"),
    (4, 3, False, "strided", "ramp"),
    (1, 1, True, "contiguous", "zero"),
    (2, 4, False, "contiguous", "ramp"),
    (5, 2, True, "strided", "large"),
]


def input_build(config):
    tokens, heads, is_neox, layout, pattern = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    width = 136 if layout == "strided" else 128
    values = torch.arange(tokens * heads * 128, dtype=torch.float32).reshape(tokens, heads, 128)
    if pattern == "ramp":
        values = ((values % 41) - 20) / 16
    elif pattern == "large":
        # Forces the e8m0 exponent up by several octaves.
        values = (values % 17 - 8) * 64
    else:
        values = torch.zeros_like(values)
    if layout == "strided":
        # Slice out of a wider BF16 buffer so dim 1 keeps a non-unit row stride.
        wide = torch.zeros((tokens, heads, width), dtype=torch.float32)
        wide[..., :128] = values
        q = wide.to(torch.bfloat16)[..., :128]
    else:
        q = values.to(torch.bfloat16)
    cache = torch.empty((6, 64), dtype=torch.float32)
    cos = [[1.0, 0.75, 0.5, 0.0, 1.0, 0.75][i] for i in range(6)]
    sin = [[0.0, 0.25, 0.5, 1.0, 0.0, 0.25][i] for i in range(6)]
    for i in range(6):
        cache[i, :32] = cos[i]
        cache[i, 32:] = sin[i]
    return {
        "positions": (torch.arange(tokens, dtype=torch.int64) * 2) % 6,
        "q": q,
        "cos_sin_cache": cache,
        "weights": (torch.arange(tokens * heads, dtype=torch.float32).reshape(tokens, heads) + 1) / 5,
        "softmax_scale": 0.25,
        "head_scale": 0.5,
        "is_neox": is_neox,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_indexer_q_rope_quant(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_indexer_q_rope_quant(**inputs)


def _to_device(inputs, device):
    """Move inputs to the device, rebuilding the wide-buffer slice for strided q."""
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    q = inputs["q"]
    if not q.is_contiguous():
        base = torch.zeros((q.shape[0], q.shape[1], q.stride(1)), dtype=q.dtype, device=device)
        base[..., : q.shape[2]] = q.to(device)
        moved["q"] = base[..., : q.shape[2]]
    return moved


def compare(ref_out, res_out, config):
    tokens, heads, _, layout, _ = config
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 2
    q_ref, w_ref = ref_out
    q_res, w_res = res_out
    assert q_res.dtype == q_ref.dtype == torch.float8_e4m3fn
    assert q_res.shape == q_ref.shape == (tokens, heads, 128)
    # Quantization is deterministic: identical FP8 bit patterns, not just values.
    assert torch.equal(q_res.view(torch.uint8), q_ref.view(torch.uint8)), "FP8 payload differs"
    assert w_res.dtype == w_ref.dtype == torch.float32
    assert w_res.shape == w_ref.shape == (tokens, heads)
    torch.testing.assert_close(w_res, w_ref, rtol=1e-6, atol=0)


def _measure(call, warmup=25, repeats=100):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


@label("v1_vllm_indexer_q_rope_quant")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_indexer_q_rope_quant(config):
    inputs = input_build(config)
    device = torch.device("cuda")
    ref_inputs = _to_device(inputs, device)
    res_inputs = _to_device(inputs, device)
    for side in (ref_inputs, res_inputs):
        assert side["q"].shape == (config[0], config[1], 128)
        assert side["q"].dtype == torch.bfloat16
        assert side["q"].is_contiguous() == (config[3] == "contiguous")
    before = res_inputs["q"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    # The kernel writes fresh outputs and never mutates its inputs.
    for output in (ref_out[0], res_out[0], ref_out[1], res_out[1]):
        assert not torch._C._is_alias_of(output, res_inputs["q"])
    assert torch.equal(res_inputs["q"], before), "q was modified in place"
    assert torch.equal(ref_inputs["q"], before), "q was modified in place"

    timed_ref = _to_device(inputs, device)
    timed_res = _to_device(inputs, device)
    ref_time = _measure(lambda: compute_ref(**timed_ref))
    res_time = _measure(lambda: compute_res(**timed_res))
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
