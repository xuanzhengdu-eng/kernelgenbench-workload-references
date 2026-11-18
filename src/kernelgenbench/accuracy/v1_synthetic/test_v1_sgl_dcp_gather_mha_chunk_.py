"""Accuracy and performance for v1_sgl_dcp_gather_mha_chunk_ (mha_chunk mode).

Operator: SGLang ``all_gather_kv_cache_for_mha_chunk_extend`` (DCP MHA
chunk-extend gather; real Gloo collective).
Official source (pinned commit 62384de1ae5e3bbd3159afd1639525b0bf39fbf9):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/dcp/comm.py#L185-L202

Contract: inputs are the nope part ``[rows, dn]`` and the RoPE part
``[rows, 1, dp]`` for this rank's share of each request's prefix, plus the
optional prefix starts.  Returns ``(nope [sum(lens), dn],
rope [sum(lens), 1, dp])`` in request/position order, both contiguous.  With DCP
disabled it returns contiguous views of the inputs without a collective.

Item: v1_sgl_dcp_gather_mha_chunk_ (723-row 639, mha_chunk mode).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import dcp_rows632_641_utils as dcp
from kernelgenbench.accuracy.v1_synthetic.rows632_641_test_utils import timings

# world, request lens, nope width, rope width, dcp_enabled, dtype, optional starts.
CASES = [
    (2, (1, 4, 7), 5, 3, True, torch.float32),
    (3, (1, 2), 4, 2, True, torch.bfloat16),
    (2, (2, 3), 3, 2, False, torch.float32),
    (2, (3, 4), 3, 2, True, torch.float32, (1, 3)),
    (2, (0, 0), 3, 2, True, torch.float32),
    (2, (0, 3), 3, 2, True, torch.float32, (1, 0)),
]
COMBOS = [(case,) for case in CASES]


def _starts(case):
    return case[6] if len(case) > 6 else (0,) * len(case[1])


def input_build(case, rank):
    """Rows pack (request, position, column); only positions owned by ``rank``
    are local.  local_rows: [n, dn], k_pe: [n, 1, dp].
    """
    world, lens, dn, dp, enabled, dtype = case[:6]
    starts = _starts(case)
    width = dn + dp
    rows = [[request * 40 + position * 3 + column for column in range(width)]
            for request, length in enumerate(lens) for position in range(length)
            if (starts[request] + position) % world == rank]
    packed = torch.tensor(rows, dtype=dtype).reshape(-1, width)
    local_rows = packed[:, :dn]
    k_pe = packed[:, dn:].unsqueeze(1)
    if not enabled:
        local_rows, k_pe = local_rows.contiguous(), k_pe.contiguous()
    inputs = {
        "local_rows": local_rows,
        "k_pe": k_pe,
        "seq_lens_cpu": torch.tensor(lens, dtype=torch.int32),
        "dcp_enabled": enabled,
    }
    if len(case) > 6:
        inputs["prefix_starts_cpu"] = torch.tensor(starts, dtype=torch.int32)
    return inputs


def expected(case, rank):
    world, lens, dn, dp, enabled, dtype = case[:6]
    starts = _starts(case)
    width = dn + dp
    rows = [[request * 40 + position * 3 + column for column in range(width)]
            for request, length in enumerate(lens) for position in range(length)
            if enabled or ((starts[request] + position) % world == rank)]
    packed = torch.tensor(rows, dtype=dtype).reshape(-1, width)
    return packed[:, :dn], packed[:, dn:].unsqueeze(1)


def compute_ref(local_rows, k_pe, seq_lens_cpu, group=None, prefix_starts_cpu=None,
                dcp_enabled=True):
    return kernelgenbench.baseline.v1_sgl_dcp_gather_mha_chunk_(
        local_rows, k_pe, seq_lens_cpu, group=group,
        prefix_starts_cpu=prefix_starts_cpu, dcp_enabled=dcp_enabled)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_dcp_gather_mha_chunk_(**inputs)


def body(rank, group, payload):
    case = payload["case"]
    inputs = input_build(case, rank)
    want_nope, want_rope = expected(case, rank)
    ref_out = compute_ref(**inputs, group=group)
    res_out = compute_res(**inputs, group=group)
    assert isinstance(res_out, tuple) and len(res_out) == 2
    assert res_out[0].shape == want_nope.shape, \
        f"nope shape {tuple(res_out[0].shape)} != {tuple(want_nope.shape)}"
    assert res_out[1].shape == want_rope.shape, \
        f"rope shape {tuple(res_out[1].shape)} != {tuple(want_rope.shape)}"
    assert res_out[0].dtype == res_out[1].dtype == case[5], "output dtype"
    assert res_out[0].is_contiguous() and res_out[1].is_contiguous(), "outputs must be contiguous"
    assert torch.equal(res_out[0], want_nope), "nope rows differ"
    assert torch.equal(res_out[1], want_rope), "rope rows differ"
    assert torch.equal(ref_out[0], want_nope) and torch.equal(ref_out[1], want_rope)
    if not case[4]:
        # DCP disabled returns contiguous views of the supplied tensors.
        assert res_out[0] is inputs["local_rows"] and res_out[1] is inputs["k_pe"]
    return dcp.time_pair(lambda: compute_ref(**inputs, group=group),
                         lambda: compute_res(**inputs, group=group))


def worker(rank, payload, rendezvous, outdir):
    dcp.dcp_worker(rank, payload, rendezvous, outdir, body)


@label("v1_sgl_dcp_gather_mha_chunk_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_dcp_gather_mha_chunk_(config):
    case, = config
    times = dcp.run(worker, {"world": case[0], "case": case})
    return timings(max(t[0] for t in times), max(t[1] for t in times))
