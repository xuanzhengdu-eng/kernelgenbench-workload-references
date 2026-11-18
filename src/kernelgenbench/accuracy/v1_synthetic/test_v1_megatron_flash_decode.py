"""Accuracy and performance for v1_megatron_flash_decode.

Operator: Megatron ``Attention.flash_decode`` (RoPE + KV-cache update + attention in
one ``flash_attn_with_kvcache`` call), pinned commit
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/attention.py#L651-L695

Contract under test
-------------------
* ``query_layer`` ``[Q, B, Hq, D]``; ``key_layer``/``value_layer``
  ``[T, B, Hkv, D]``; ``inference_key_memory``/``inference_value_memory``
  ``[C, B, Hkv, D]``; ``sequence_len_offset`` ``[B]`` int32; ``Hq`` divisible by
  ``Hkv``.
* Rotated features: ``2 * rotary_cos.shape[-1]``; ``rotary_cos``/``rotary_sin``
  are ``[R, Dr/2]`` or ``None``.  ``rotary_interleaved`` switches between the
  split-half and the pair layout.
* Every query position of batch ``b`` rotates at the batch offset; new key rows
  rotate at ``offset + t``.  The caches are updated in place at
  ``[offset, offset + T)``; bytes outside that window are untouched.
* Return is ``[B, Q, Hq, D]`` in the query dtype, freshly allocated, and no
  backward is defined.
* ``flash_attn_with_kvcache`` defaults apply: noncausal, no local window,
  scale ``1/sqrt(D)``.

Target backend status: the pinned Megatron asserts that
``flash_attn_with_kvcache`` is importable.  The pinned containers here have no
flash-attn build (``ModuleNotFoundError: No module named 'flash_attn'``;
``flash_attn_interface``/``flashinfer`` absent), so this backend is source-checked
but not executed on this host.

Item: v1_megatron_flash_decode (723-row 127).
"""
import math

import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rope_rows122_131_checks as checks


# Axes: (Q, B, Hq, Hkv, D, T, cache capacity C, offsets, rotary dim Dr,
#        interleaved, dtype).
COMBOS = [
    (1, 2, 4, 2, 32, 1, 8, (0, 3), 32, False, torch.float16),
    (2, 2, 4, 1, 32, 2, 8, (1, 3), 16, True, torch.bfloat16),
    (3, 1, 2, 2, 32, 2, 8, (4,), 0, False, torch.float16),
    (2, 1, 2, 1, 64, 3, 8, (2,), 32, False, torch.bfloat16),
    (1, 1, 1, 1, 32, 1, 6, (5,), 16, True, torch.float16),
]

TOLERANCE = {torch.float16: (1.2e-2, 3e-3), torch.bfloat16: (3e-2, 1e-2)}


def input_build(config):
    """Deterministic trigonometric inputs; cos/sin tables are [C + T, Dr/2]."""
    (q_len, batch, heads, kv_heads, head_dim, new_len, capacity, offsets,
     rotary_dim, interleaved, dtype) = config
    generator = torch.Generator(device="cpu").manual_seed(20261227)

    def make(shape, phase):
        flat = torch.arange(1, 1 + int(torch.tensor(shape).prod()), dtype=torch.float32)
        return ((flat * 0.17 + phase).sin() * 0.8).reshape(shape).to(dtype)

    query = make((q_len, batch, heads, head_dim), 0.2)
    key = make((new_len, batch, kv_heads, head_dim), 0.7)
    value = make((new_len, batch, kv_heads, head_dim), 1.1)
    key_cache = make((capacity, batch, kv_heads, head_dim), 2.0)
    value_cache = make((capacity, batch, kv_heads, head_dim), 2.9)
    if rotary_dim:
        theta = (torch.arange((capacity + new_len) * (rotary_dim // 2), dtype=torch.float32)
                 .reshape(capacity + new_len, rotary_dim // 2) * 0.071)
        cos, sin = theta.cos(), theta.sin()
    else:
        cos = sin = None
    return {
        "sequence_len_offset": torch.tensor(offsets, dtype=torch.int32),
        "query_layer": query,
        "key_layer": key,
        "value_layer": value,
        "inference_key_memory": key_cache,
        "inference_value_memory": value_cache,
        "rotary_cos": cos,
        "rotary_sin": sin,
        "rotary_interleaved": interleaved,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_flash_decode(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_flash_decode(**inputs)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, torch.Tensor), f"expected a tensor, got {type(res_out)}"
    assert res_out.shape == ref_out.shape, \
        f"shape {tuple(res_out.shape)} != {tuple(ref_out.shape)}"
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = TOLERANCE[dtype or ref_out.dtype]
    torch.testing.assert_close(res_out.float(), ref_out.float(), rtol=rtol, atol=atol)


def _oracle(inputs):
    """Independent FP64 scalar oracle; every rotation is rounded through the input
    dtype exactly like the kernel does, then accumulated in float64.
    """
    query = inputs["query_layer"].double()
    key = inputs["key_layer"].double()
    value = inputs["value_layer"].double()
    key_cache = inputs["inference_key_memory"].clone()
    value_cache = inputs["inference_value_memory"].clone()
    offsets = inputs["sequence_len_offset"].tolist()
    q_len, batch, heads, head_dim = query.shape
    new_len, _, kv_heads, _ = key.shape
    cos = None if inputs["rotary_cos"] is None else inputs["rotary_cos"].to(
        inputs["query_layer"].dtype).double()
    sin = None if cos is None else inputs["rotary_sin"].to(
        inputs["query_layer"].dtype).double()
    interleaved = inputs["rotary_interleaved"]

    def rotate(vector, position):
        out = vector.clone()
        if cos is None:
            return out
        half = cos.shape[-1]
        for j in range(half):
            a, b = (2 * j, 2 * j + 1) if interleaved else (j, half + j)
            x0, x1 = float(vector[a]), float(vector[b])
            c, s = float(cos[position, j]), float(sin[position, j])
            out[a] = x0 * c - x1 * s
            out[b] = x0 * s + x1 * c
        return out

    result = torch.empty((batch, q_len, heads, head_dim), dtype=query.dtype)
    for b in range(batch):
        start = offsets[b]
        for t in range(new_len):
            for h in range(kv_heads):
                key_cache[start + t, b, h] = rotate(key[t, b, h], start + t)
            value_cache[start + t, b] = value[t, b]
        for t in range(q_len):
            for h in range(heads):
                q = rotate(query[t, b, h], start)
                kv_head = h // (heads // kv_heads)
                logits = [sum(float(q[j]) * float(key_cache[i, b, kv_head, j])
                              for j in range(head_dim)) / math.sqrt(head_dim)
                          for i in range(start + new_len)]
                peak = max(logits)
                weights = [math.exp(z - peak) for z in logits]
                total = sum(weights)
                for j in range(head_dim):
                    result[b, t, h, j] = sum(
                        w * float(value_cache[i, b, kv_head, j])
                        for i, w in enumerate(weights)) / total
    return result.to(inputs["query_layer"].dtype), key_cache.to(
        inputs["query_layer"].dtype), value_cache.to(inputs["query_layer"].dtype)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_megatron_flash_decode, inputs,
                          kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_megatron_flash_decode, inputs,
                          kernelgenbench.device)


@label("v1_megatron_flash_decode")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_flash_decode(config):
    (q_len, batch, heads, kv_heads, head_dim, new_len, capacity, offsets,
     rotary_dim, interleaved, dtype) = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    source = checks.fresh(inputs, torch.device("cpu"))

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype=dtype)

    expected, expected_key, expected_value = _oracle(source)
    rtol, atol = TOLERANCE[dtype]
    for label_name, produced, side in (("reference", ref_out, ref_inputs),
                                       ("solution", res_out, res_inputs)):
        assert produced.shape == (batch, q_len, heads, head_dim), label_name
        torch.testing.assert_close(produced.float().cpu(), expected.float(),
                                   rtol=rtol, atol=atol)
        torch.testing.assert_close(side["inference_key_memory"].float().cpu(),
                                   expected_key.float(), rtol=5e-3, atol=2e-3)
        torch.testing.assert_close(side["inference_value_memory"].float().cpu(),
                                   expected_value.float(), rtol=0, atol=0)
        # Only the [offset, offset + new_len) window of each batch item moves.
        for b, start in enumerate(offsets):
            end = start + new_len
            for name in ("inference_key_memory", "inference_value_memory"):
                assert torch.equal(side[name][:start, b].cpu(), inputs[name][:start, b])
                assert torch.equal(side[name][end:, b].cpu(), inputs[name][end:, b])
        # The read-only inputs stay untouched.
        for name in ("query_layer", "key_layer", "value_layer", "sequence_len_offset",
                     "rotary_cos", "rotary_sin"):
            if inputs[name] is not None:
                torch.testing.assert_close(side[name].cpu(), inputs[name], rtol=0, atol=0)
        # The return value never aliases an input or a cache.
        for name, tensor in side.items():
            if isinstance(tensor, torch.Tensor) and tensor.numel():
                assert not torch._C._is_alias_of(produced, tensor), f"output aliases {name}"

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
