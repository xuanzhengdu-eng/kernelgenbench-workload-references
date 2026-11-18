"""Accuracy and performance for v1_vllm_ultraquant_store_.

Packs BF16/FP16 K (Hadamard-rotated) and V into the UltraQuant AoS FP4 cache:
per 32-element group one UE8M0 scale byte and 16 packed E2M1 nibbles. Padding
slots (<0) and bytes outside the D + D/16 data region must stay untouched.
"""
import torch

import kernelgenbench
import statistics
import time
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# tokens, heads, head_dim, block size, dtype, slot padding, zero token, constant c
COMBOS = [
    (4, 2, 32, 4, torch.float16, 7, False, .156),
    (5, 1, 64, 4, torch.bfloat16, 13, True, .156),
    (7, 3, 128, 4, torch.float16, 0, True, .2),
    (3, 1, 32, 4, torch.bfloat16, 9, True, .156),
]


def get_input(case, device=None):
    device = kernelgenbench.device if device is None else device
    n, h, d, bs, dtype, pad, zero, c = case
    gen = torch.Generator().manual_seed(101 + d + n)
    k = torch.randn(n, h, d, generator=gen, dtype=torch.float32).to(dtype).to(device)
    v = torch.randn(n, h, d, generator=gen, dtype=torch.float32).to(dtype).to(device)
    if zero:
        k[1].zero_()
        v[1].zero_()
    if n == 3:
        pattern = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., -.25, -.75, -1.25,
                                -1.75, -2.5, -3.5, -5., 0., 4.], dtype=dtype, device=device)
        v[1] = pattern.repeat(d // len(pattern)).reshape(1, d)
    slots = torch.arange(n, device=device, dtype=torch.int64)
    slots[0] = -1
    # +1 block keeps every slot below the allocation bound and leaves tail bytes.
    cache = torch.full(((n + bs - 1) // bs + 1, bs, h, d + d // 16 + pad), 173,
                       dtype=torch.uint8, device=device)
    return dict(key=k, value=v, kv_cache=cache, slot_mapping=slots, constant_c=c)


def measure(fn, inputs):
    """Median single-call ms; input copies and warmup stay out of the timed region."""
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


def input_build(case):
    return get_input(case)


def compute_ref(**kw):
    return kernelgenbench.baseline.v1_vllm_ultraquant_store_(**kw)


def compute_res(**kw):
    return kernelgenbench.triton.v1_vllm_ultraquant_store_(**kw)


def oracle(case):
    """Scalar double Hadamard and scalar table quantizer, independent of reference."""
    n, h, d, bs, _, _, _, c = case
    a = get_input(case, device="cpu")
    out = a["kv_cache"].clone()
    levels = [-6., -4., -3., -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 3., 4., 6.]
    mids = [(levels[i] + levels[i + 1]) / 2 for i in range(14)]

    def pack(x):
        codes = []
        scales = []
        for off in range(0, d, 32):
            g = x[off:off + 32]
            m = max(map(abs, g))
            e = int(__import__("math").floor(__import__("math").log2(c * m) + .5)) if m else 0
            scales.append(e + 127 if m else 0)
            sc = 2. ** e if m else 1
            for val in g:
                ix = sum(val / sc > bound for bound in mids)
                codes.append(15 - ix if ix < 7 else ix - 7)
        return [codes[i] | (codes[i + 1] << 4) for i in range(0, d, 2)], scales

    for t in range(1, n):
        for head in range(h):
            k = a["key"][t, head].float().tolist()
            v = a["value"][t, head].float().tolist()
            for j in range(d.bit_length() - 1):
                bit = 1 << j
                old = k[:]
                k = [((old[i] if not (i & bit) else -old[i]) + old[i ^ bit]) * 2 ** -.5
                     for i in range(d)]
            kc, ks = pack(k)
            vc, vs = pack(v)
            out[t // bs, t % bs, head, :d + d // 16] = torch.tensor(kc + ks + vc + vs,
                                                                   dtype=torch.uint8)
    return out


def compute_ref_performance(**kw):
    return measure(compute_ref, kw)


def compute_res_performance(**kw):
    return measure(compute_res, kw)


@label("v1_vllm_ultraquant_store_")
@parametrize("case", COMBOS)
def test_accuracy_v1_vllm_ultraquant_store_(case):
    a = get_input(case)
    b = get_input(case)
    expected = oracle(case)
    before = [b[x].clone() for x in ("key", "value", "slot_mapping")]
    ptr = b["kv_cache"].data_ptr()
    assert compute_ref(**a) is None
    assert compute_res(**b) is None
    assert ptr == b["kv_cache"].data_ptr()
    assert torch.equal(a["kv_cache"].cpu(), expected)
    assert torch.equal(b["kv_cache"].cpu(), expected)
    for key, saved in zip(("key", "value", "slot_mapping"), before):
        assert torch.equal(b[key], saved), key
    ref_time = compute_ref_performance(**a)
    res_time = compute_res_performance(**b)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
