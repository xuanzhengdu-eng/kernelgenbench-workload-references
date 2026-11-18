"""Accuracy and performance for v1_sgl_dcp_gather_mha_extend_ (mha_extend mode).

Operator: SGLang ``all_gather_kv_cache_for_mha_extend`` (DCP MHA extend gather;
real Gloo collective).
Official source (pinned commit 62384de1ae5e3bbd3159afd1639525b0bf39fbf9):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/dcp/comm.py#L204-L256

Contract: the stored prefix rows are supplied in cache order with
``local_indices`` giving the read order; the entry gathers the cached prefix with
``all_gather_kv_cache_for_dcp``, splits nope/RoPE, casts the prefix to the
extend dtype, then interleaves per request as ``prefix rows of request r`` then
``extend rows of request r``.  Returns ``(nope [tokens, dn], rope [tokens, 1, dp])``,
both contiguous, where ``tokens = sum(prefix lens + extend lens)``.

Item: v1_sgl_dcp_gather_mha_extend_ (723-row 639, mha_extend mode).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import dcp_rows632_641_utils as dcp
from kernelgenbench.accuracy.v1_synthetic.rows632_641_test_utils import timings

# world, prefix lens, extend lens, nope width, rope width, extend dtype, cache dtype.
CASES = [
    (2, (2, 4), (1, 2), 3, 2, torch.float32, torch.bfloat16),
    (3, (1, 2), (2, 1), 4, 3, torch.bfloat16, torch.bfloat16),
    (2, (0, 2), (2, 0), 3, 2, torch.float32, torch.bfloat16),
]
COMBOS = [(case,) for case in CASES]


def _prefix_rows(case, rank):
    world, lens, _ext, dn, dp, _dtype, _cache_dtype = case
    width = dn + dp
    return [[request * 40 + position * 3 + column for column in range(width)]
            for request, length in enumerate(lens) for position in range(length)
            if position % world == rank]


def input_build(case, rank):
    """Prefix rows are stored flipped with ``local_indices`` unflipping them;
    extend rows carry positions ``lens[req] + pos``.  Shapes: local_rows
    [n, 1, dn], k_pe [n, 1, dp], extend_nope [m, dn], extend_rope [m, 1, dp].
    """
    world, lens, ext, dn, dp, dtype, cache_dtype = case
    width = dn + dp
    prefix = torch.tensor(_prefix_rows(case, rank), dtype=cache_dtype).reshape(-1, 1, width)
    stored = prefix.flip(0).contiguous()
    local_indices = torch.arange(prefix.shape[0] - 1, -1, -1, dtype=torch.int64)
    newer = [[request * 40 + (lens[request] + position) * 3 + column
              for column in range(width)]
             for request, count in enumerate(ext) for position in range(count)]
    extend = torch.tensor(newer, dtype=dtype).reshape(-1, 1, width)
    return {
        "local_rows": stored[:, :, :dn],
        "k_pe": stored[:, :, dn:],
        "seq_lens_cpu": torch.tensor(lens, dtype=torch.int32),
        "extend_nope": extend[:, :, :dn].squeeze(1),
        "extend_rope": extend[:, :, dn:],
        "extend_lens_cpu": torch.tensor(ext, dtype=torch.int32),
        "local_indices": local_indices,
    }


def expected(case, rank):
    """Full gathered sequence: every request's positions 0..n-1, then its extends."""
    _world, lens, ext, dn, dp, dtype, _cache_dtype = case
    width = dn + dp
    rows = []
    for request, length in enumerate(lens):
        rows.extend([request * 40 + position * 3 + column for column in range(width)]
                    for position in range(length))
        rows.extend([request * 40 + (length + position) * 3 + column
                     for column in range(width)] for position in range(ext[request]))
    packed = torch.tensor(rows, dtype=dtype).reshape(-1, width)
    return packed[:, :dn], packed[:, dn:].unsqueeze(1)


def compute_ref(local_rows, k_pe, seq_lens_cpu, extend_nope, extend_rope,
                extend_lens_cpu, group=None, local_indices=None):
    return kernelgenbench.baseline.v1_sgl_dcp_gather_mha_extend_(
        local_rows, k_pe, seq_lens_cpu, extend_lens_cpu, extend_nope,
        extend_rope, group=group, local_indices=local_indices)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_dcp_gather_mha_extend_(**inputs)


def body(rank, group, payload):
    case = payload["case"]
    inputs = input_build(case, rank)
    want_nope, want_rope = expected(case, rank)
    ref_out = compute_ref(**inputs, group=group)
    res_out = compute_res(**inputs, group=group)
    assert isinstance(res_out, tuple) and len(res_out) == 2
    assert res_out[0].shape == want_nope.shape, "nope shape"
    assert res_out[1].shape == want_rope.shape, "rope shape"
    assert res_out[0].dtype == want_nope.dtype, "prefix must be cast to the extend dtype"
    assert res_out[1].dtype == want_rope.dtype
    assert res_out[0].is_contiguous() and res_out[1].is_contiguous()
    assert torch.equal(res_out[0], want_nope), "prefix/extend nope order differs"
    assert torch.equal(res_out[1], want_rope), "prefix/extend rope order differs"
    assert torch.equal(ref_out[0], want_nope) and torch.equal(ref_out[1], want_rope)
    return dcp.time_pair(lambda: compute_ref(**inputs, group=group),
                         lambda: compute_res(**inputs, group=group))


def worker(rank, payload, rendezvous, outdir):
    dcp.dcp_worker(rank, payload, rendezvous, outdir, body)


@label("v1_sgl_dcp_gather_mha_extend_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_dcp_gather_mha_extend_(config):
    case, = config
    times = dcp.run(worker, {"world": case[0], "case": case})
    return timings(max(t[0] for t in times), max(t[1] for t in times))
