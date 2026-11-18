"""FA2 paged prefill/decode: local COMBOS, scalar oracle, side effects and timing.

Official entry: Megatron ``Attention.flash_decode_and_prefill`` (varlen causal
prefill via ``flash_attn_varlen_func``, paged decode via
``flash_attn_with_kvcache``). The pinned runtime has neither kernel installed,
so the H800 run is recorded as pending in report.csv / decision.json; the local
reference and this test still pin the full input/output contract.
"""
import math
import time

import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult

# query lengths, key lengths, Hq, Hkv, D, decode-only, custom scale, dtype
COMBOS = [
    ((1, 3), (3, 4), 4, 2, 32, False, None, torch.float16),
    ((4, 2), (2, 4), 2, 1, 32, False, 0.7, torch.bfloat16),
    ((2, 2), (3, 5), 4, 1, 32, True, 0.7, torch.float16),
    ((1, 1), (257, 3), 2, 1, 32, True, None, torch.bfloat16),
    ((1, 2), (257, 2), 1, 1, 32, False, 0.11, torch.float16),
]


def input_build(config):
    """CPU inputs: q [sum(q_lens),1,Hq,D], k/v [pages,256,Hkv,D], block table.

    Pages are 256 slots wide. Causal masking is bottom-right aligned, so a
    query may be longer than its key length; such leading queries are zero.
    """
    ql, kl, heads, kv_heads, dim, decode, scale, dtype = config
    batch = len(ql)
    pages = [(x + 255) // 256 for x in kl]
    total = sum(pages)
    width = max(pages)
    table = torch.full((batch, width), -1, dtype=torch.int32)
    page = 0
    for b, count in enumerate(pages):
        table[b, :count] = torch.tensor([total - 1 - j for j in range(page, page + count)],
                                        dtype=torch.int32)
        page += count

    def tensor(shape, phase):
        n = 1
        for z in shape:
            n *= z
        return ((torch.arange(n, dtype=torch.float32) * 0.019 + phase).sin() * 0.8).reshape(shape).to(dtype)

    def cumulative(lengths):
        out = [0]
        for x in lengths:
            out.append(out[-1] + x)
        return torch.tensor(out, dtype=torch.int32)

    return dict(q=tensor((sum(ql), 1, heads, dim), 0.2),
                k=tensor((total, 256, kv_heads, dim), 1.4),
                v=tensor((total, 256, kv_heads, dim), 2.4),
                max_seqlen_q=max(ql), max_seqlen_k=max(kl),
                cu_seqlens_q=cumulative(ql), cu_seqlens_k=cumulative(kl),
                seqlens_k=torch.tensor(kl, dtype=torch.int32),
                block_table=table, is_decode_only=decode, softmax_scale=scale)


def _to_device(inputs, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}


def compute_ref(**kw):
    return kernelgenbench.baseline.v1_megatron_paged_attn(**kw)


def compute_res(**kw):
    return kernelgenbench.solution.v1_megatron_paged_attn(**kw)


def scalar_oracle(d):
    """Independent per-token scalar attention over gathered pages."""
    q = d["q"].cpu()
    k = d["k"].cpu()
    v = d["v"].cpu()
    table = d["block_table"].cpu().tolist()
    cu = d["cu_seqlens_q"].cpu().tolist()
    lens = d["seqlens_k"].cpu().tolist()
    heads, dim = q.shape[2], q.shape[3]
    kv_heads = k.shape[2]
    out = torch.zeros_like(q)
    # Decode always uses the default scale; prefill honours an explicit one.
    scale = dim ** -0.5 if d["max_seqlen_q"] <= 1 or d["softmax_scale"] is None else d["softmax_scale"]
    for b, kn in enumerate(lens):
        lo, hi = cu[b:b + 2]
        qlen = hi - lo
        for qi in range(qlen):
            count = max(0, min(kn, qi + kn - qlen + 1))
            if count == 0:
                continue
            for h in range(heads):
                kh = h // (heads // kv_heads)
                query = [float(x) for x in q[lo + qi, 0, h]]
                rows = [(table[b][i // 256], i % 256) for i in range(count)]
                logits = [sum(query[j] * float(k[page, pos, kh, j]) for j in range(dim)) * scale
                          for page, pos in rows]
                mx = max(logits)
                ws = [math.exp(x - mx) for x in logits]
                den = sum(ws)
                for j in range(dim):
                    out[lo + qi, 0, h, j] = sum(w * float(v[page, pos, kh, j])
                                                for w, (page, pos) in zip(ws, rows)) / den
    return out


def compare(ref_out, res_out, config=None, inputs=None):
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == inputs["q"].shape
    assert res_out.dtype == ref_out.dtype == config[7]
    assert res_out.device == ref_out.device == inputs["q"].device
    assert res_out.is_contiguous()
    for value in inputs.values():
        if isinstance(value, torch.Tensor):
            assert not torch._C._is_alias_of(res_out, value)
    expected = scalar_oracle(inputs).to(res_out.device)
    torch.testing.assert_close(res_out, expected, rtol=0.018, atol=0.004)
    torch.testing.assert_close(ref_out, expected, rtol=0.018, atol=0.004)
    # Queries whose causal window contains no key stay exactly zero.
    cu = inputs["cu_seqlens_q"].cpu().tolist()
    lens = inputs["seqlens_k"].cpu().tolist()
    for b in range(len(lens)):
        missing = max(0, cu[b + 1] - cu[b] - lens[b])
        assert torch.count_nonzero(res_out[cu[b]:cu[b] + missing]) == 0


def compute_ref_performance(**inputs):
    return _bench(compute_ref, inputs)


def compute_res_performance(**inputs):
    return _bench(compute_res, inputs)


def _bench(fn, inputs):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(3):
        fn(**inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000 / 3


@label("v1_megatron_paged_attn")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_paged_attn(config):
    base = input_build(config)
    ref_inputs = _to_device(base, kernelgenbench.device)
    res_inputs = _to_device(base, kernelgenbench.device)
    saved = {k: v.clone() for k, v in res_inputs.items() if isinstance(v, torch.Tensor)}

    want = compute_ref(**ref_inputs)
    got = compute_res(**res_inputs)
    compare(want, got, config, res_inputs)

    # Every input buffer is read-only for this entry.
    for k, v in saved.items():
        assert torch.equal(v, res_inputs[k]), f"input {k} was modified"

    ta = compute_ref_performance(**ref_inputs)
    tb = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(ref_time=ta, res_time=tb, speedup=ta / tb if tb else float("inf"))
