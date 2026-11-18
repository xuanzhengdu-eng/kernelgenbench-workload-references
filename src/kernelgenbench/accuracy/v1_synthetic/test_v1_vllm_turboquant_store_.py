"""Accuracy and performance for the AoS v1_vllm_turboquant_store_ entry.

Two data paths of the official AoS launcher are covered: FP8 E4M3 keys with
3/4-bit uniform V, and MSE keys (normalise -> rotate by PiT -> bucketize by
midpoints -> fp16 norm) with the same uniform V quantizer.  Cases include a
constant V row, a zero K/V token, cache padding, an untouched padding slot and
a custom key layout gap.
"""
import math
import statistics
import struct
import time
from functools import lru_cache

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# N, H, D, block, dtype, V bits, slot padding, constant V
FP8_CASES = [
    (4, 2, 32, 4, torch.float16, 3, 5, False),
    (5, 1, 64, 4, torch.bfloat16, 4, 8, True),
    (7, 3, 128, 4, torch.float16, 3, 0, True),
    (3, 2, 64, 4, torch.float16, 4, 3, False),
]
# N, H, D, block, dtype, K bits, V bits, key gap, cache padding, zero K/const V, permuted PiT
MSE_CASES = [
    (4, 2, 32, 4, torch.float16, 3, 3, 0, 5, False, False),
    (5, 1, 64, 4, torch.bfloat16, 4, 4, 3, 8, True, True),
    (7, 3, 128, 4, torch.float16, 3, 4, 2, 0, True, True),
    (3, 2, 64, 4, torch.float16, 4, 3, 0, 3, True, False),
]

COMBOS = [('fp8', c) for c in FP8_CASES] + [('mse', c) for c in MSE_CASES]


@lru_cache(maxsize=None)
def e4m3(x):
    """Enumerate positive finite E4M3FN encodings, nearest with ties to even."""
    sign = 128 if math.copysign(1, x) < 0 else 0
    u = abs(x)

    def val(b):
        e, m = b >> 3, b & 7
        return 2 ** -6 * m / 8 if not e else 2 ** (e - 7) * (1 + m / 8)

    return min(range(127), key=lambda b: (abs(u - val(b)), b & 1)) + sign


def pack(q, bits):
    if bits == 4:
        return [q[i] | q[i + 1] << 4 for i in range(0, len(q), 2)]
    out = []
    for i in range(0, len(q), 8):
        w = sum(q[i + r] << (3 * r) for r in range(8))
        out.extend([w & 255, (w >> 8) & 255, (w >> 16) & 255])
    return out


def input_build(case):
    """CPU inputs for one (mode, case) pair; the cache is a full uint8 slot."""
    mode, c = case
    device = "cpu"
    if mode == 'fp8':
        n, h, d, bs, dtype, bits, pad, constant = c
        gen = torch.Generator().manual_seed(905 + n + d + bits)
        key = (torch.randn(n, h, d, generator=gen) * 1.9).to(dtype).to(device)
        value = (torch.randn(n, h, d, generator=gen) * 1.9).to(dtype).to(device)
        if constant:
            value[1].fill_(1.25)
        if n == 3:
            kp = torch.tensor([0., .0009765625, -.0009765625, 1.0625, 1.1875, -1.0625,
                               -1.1875, 2., -2., 3., -3., 4., -4., .5, -.5, 1.], dtype=dtype)
            vp = torch.tensor([0., .5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5,
                               11.5, 12.5, 13.5, 15.], dtype=dtype)
            key[1] = kp.repeat(d // 16).expand(h, d)
            value[1] = vp.repeat(d // 16).expand(h, d)
        cache = torch.full(((n + bs - 1) // bs + 1, bs, h, d + d * bits // 8 + 4 + pad),
                           173, dtype=torch.uint8)
        slots = torch.arange(n, dtype=torch.int32)
        slots[0] = -1
        return dict(key=key, value=value, kv_cache=cache, slot_mapping=slots,
                    PiT=None, midpoints=None, mse_bits=3, key_packed_size=d,
                    value_quant_bits=bits, key_fp8=True)
    n, h, d, bs, dtype, kbits, vbits, gap, pad, zero, perm = c
    gen = torch.Generator().manual_seed(152 + n + d + kbits + vbits)
    key = (torch.randn(n, h, d, generator=gen) * 1.9).to(dtype)
    value = (torch.randn(n, h, d, generator=gen) * 1.9).to(dtype)
    if zero:
        key[1].zero_()
        value[1].fill_(1.25)
    pit = torch.eye(d, dtype=torch.float32)
    if perm:
        pit = pit[:, torch.arange(d - 1, -1, -1)]
    # Midpoint 0 is exactly representable, so bucketize must use >= at zero.
    # The official kernel reads n_centroids-1 = 2**kbits-1 midpoints (see the
    # `Midpoints_ptr  # [n_centroids-1]` contract), so build exactly that many.
    mids = (torch.arange(2 ** kbits - 1, dtype=torch.float32)
            - (2 ** kbits - 2) / 2) * .0625
    kps = d * kbits // 8 + 2 + gap
    cache = torch.full(((n + bs - 1) // bs + 1, bs, h, kps + d * vbits // 8 + 4 + pad),
                       173, dtype=torch.uint8)
    slots = torch.arange(n, dtype=torch.int32)
    slots[0] = -1
    return dict(key=key, value=value, kv_cache=cache, slot_mapping=slots, PiT=pit,
                midpoints=mids, mse_bits=kbits, key_packed_size=kps,
                value_quant_bits=vbits, key_fp8=False)


def oracle(case):
    """Scalar double-precision expectation, independent of the reference."""
    mode, c = case
    a = input_build(case)
    out = a['kv_cache'].clone()
    if mode == 'fp8':
        n, h, d, bs, _, bits, _, _ = c
        for t in range(1, n):
            for j in range(h):
                k = a['key'][t, j].float().tolist()
                v = a['value'][t, j].float().tolist()
                mn = min(v)
                scale = max((max(v) - mn) / (2 ** bits - 1), 1e-8)
                q = [min(max(int((x - mn) / scale + .5), 0), 2 ** bits - 1) for x in v]
                raw = bytes([e4m3(x) for x in k] + pack(q, bits)) + struct.pack('<ee', scale, mn)
                out[t // bs, t % bs, j, :len(raw)] = torch.tensor(list(raw), dtype=torch.uint8)
        return out
    n, h, d, bs, _, kbits, vbits, gap, _, _, perm = c
    mid = a['midpoints'].tolist()
    kb = d * kbits // 8
    kps = kb + 2 + gap
    for t in range(1, n):
        for j in range(h):
            k = a['key'][t, j].float().tolist()
            v = a['value'][t, j].float().tolist()
            norm = math.sqrt(sum(x * x for x in k))
            y = [x / (norm + 1e-8) for x in k]
            if perm:
                y = y[::-1]
            kq = [sum(x >= m for m in mid) for x in y]
            mn = min(v)
            scale = max((max(v) - mn) / (2 ** vbits - 1), 1e-8)
            vq = [min(max(int((x - mn) / scale + .5), 0), 2 ** vbits - 1) for x in v]
            raw = out[t // bs, t % bs, j]
            raw[:kb] = torch.tensor(pack(kq, kbits), dtype=torch.uint8)
            raw[kb:kb + 2] = torch.tensor(list(struct.pack('<e', norm)), dtype=torch.uint8)
            vb = bytes(pack(vq, vbits)) + struct.pack('<ee', scale, mn)
            raw[kps:kps + len(vb)] = torch.tensor(list(vb), dtype=torch.uint8)
    return out


def measure(fn, inputs):
    """Median single-call ms; input copies and warmup stay outside the timed region."""
    samples = []
    for _ in range(6):
        arguments = {name: value.to(device=kernelgenbench.device, copy=True)
                     if isinstance(value, torch.Tensor) else value
                     for name, value in inputs.items()}
        if kernelgenbench.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(**arguments)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            fn(**arguments)
            samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples[1:])


def _call(fn, kw):
    return fn(**{name: value for name, value in kw.items() if name != "layout"})


def compute_ref(**kw):
    return _call(kernelgenbench.baseline.v1_vllm_turboquant_store_, kw)


def compute_res(**kw):
    return kernelgenbench.triton.v1_vllm_turboquant_store_(**kw)


def compute_ref_performance(**kw):
    return measure(compute_ref, kw)


def compute_res_performance(**kw):
    return measure(compute_res, kw)


@label("v1_vllm_turboquant_store_")
@label("v1_vllm_turboquant_store_::vllm_aos")
@parametrize("case", COMBOS)
@pytest.mark.parametrize("case", COMBOS)
def test_accuracy_v1_vllm_turboquant_store_(case):
    mode, _ = case
    cpu = input_build(case)
    side_a, side_b = {}, {}
    for side in (side_a, side_b):
        for name, value in cpu.items():
            side[name] = (value.to(kernelgenbench.device) if isinstance(value, torch.Tensor)
                          else value)
    expected = oracle(case).to(kernelgenbench.device)
    saved = {k: v.clone() for k, v in side_b.items()
             if k not in ('kv_cache',) and isinstance(v, torch.Tensor)}
    ptr = side_b['kv_cache'].data_ptr()

    assert compute_ref(**side_a) is None
    assert compute_res(**side_b) is None
    torch.cuda.synchronize()

    assert side_b['kv_cache'].data_ptr() == ptr
    assert torch.equal(side_a['kv_cache'], expected), mode
    assert torch.equal(side_b['kv_cache'], expected), mode
    for name, value in saved.items():
        assert torch.equal(side_b[name], value), name
    # Padding slots keep the pre-run sentinel byte-for-byte.
    bs = side_b['kv_cache'].shape[1]
    for t, slot in enumerate(side_b['slot_mapping'].tolist()):
        if slot < 0:
            assert torch.equal(side_b['kv_cache'][slot // bs, slot % bs],
                               cpu['kv_cache'][slot // bs, slot % bs].to(
                                   kernelgenbench.device)), t

    ref_time = compute_ref_performance(**cpu)
    res_time = compute_res_performance(**cpu)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
