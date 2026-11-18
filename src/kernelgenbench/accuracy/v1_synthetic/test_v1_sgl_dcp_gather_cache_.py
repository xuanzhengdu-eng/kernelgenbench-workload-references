"""Accuracy and performance for v1_sgl_dcp_gather_cache_ (dcp_cache mode).

Operator: SGLang ``all_gather_kv_cache_for_dcp`` (cached-prefix DCP gather; real
Gloo collective).
Official source (pinned commit 62384de1ae5e3bbd3159afd1639525b0bf39fbf9):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/dcp/comm.py#L337-L437
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/dcp/comm.py#L162-L183

Contract: ``local_rows``/``k_pe`` are ``[rows, 1, dn]``/``[rows, 1, dp]`` for
this rank's share of each request's cached prefix; ``prefix_starts_cpu`` offsets
the request inside the DCP cycle and ``interleave_size`` assigns that many
consecutive positions per rank.  The packed result is ``[sum(lens), 1, dn+dp]``
in request/position order.  With DCP disabled the local packed rows are returned
unchanged.  Inputs are read-only.

Item: v1_sgl_dcp_gather_cache_ (723-row 639, cache mode).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import dcp_rows632_641_utils as dcp
from kernelgenbench.accuracy.v1_synthetic.rows632_641_test_utils import timings

# world, request lens, prefix starts, nope width, rope width, interleave, dtype,
# optional dcp_enabled=False last element.
CASES = [
    (2, (5, 6), (1, 3), 4, 2, 2, torch.float32),
    (3, (2, 7), (2, 5), 3, 2, 1, torch.bfloat16),
    (2, (3, 8), (3, 7), 3, 1, 4, torch.float32),
    (2, (3, 4), (1, 2), 3, 2, 2, torch.float32, False),
    (2, (0, 3), (1, 0), 3, 2, 2, torch.float32),
    (2, (0, 0), (0, 0), 3, 2, 1, torch.float32),
]
COMBOS = [(case,) for case in CASES]


def _own(position, start, interleave, world, rank):
    return ((start + position) // interleave) % world == rank


def input_build(case, rank):
    """Packed rows carry (request, position, column) in their values; only
    positions that map to ``rank`` are local.  Shape [rows, 1, dn+dp].
    """
    world, lens, starts, dn, dp, interleave, dtype = case[:7]
    width = dn + dp
    rows = [[request * 40 + position * 3 + column for column in range(width)]
            for request, length in enumerate(lens) for position in range(length)
            if _own(position, starts[request], interleave, world, rank)]
    packed = torch.tensor(rows, dtype=dtype).reshape(-1, 1, width)
    return {
        "local_rows": packed[:, :, :dn],
        "k_pe": packed[:, :, dn:],
        "seq_lens_cpu": torch.tensor(lens, dtype=torch.int32),
        "prefix_starts_cpu": torch.tensor(starts, dtype=torch.int32),
        "interleave_size": interleave,
        "dcp_enabled": case[7] if len(case) > 7 else True,
    }


def expected(case, rank):
    world, lens, starts, dn, dp, interleave, dtype = case[:7]
    enabled = case[7] if len(case) > 7 else True
    width = dn + dp
    rows = [[request * 40 + position * 3 + column for column in range(width)]
            for request, length in enumerate(lens) for position in range(length)
            if enabled or _own(position, starts[request], interleave, world, rank)]
    return torch.tensor(rows, dtype=dtype).reshape(-1, 1, width)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_dcp_gather_cache_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_dcp_gather_cache_(**inputs)


def body(rank, group, payload):
    case = payload["case"]
    inputs = input_build(case, rank)
    want = expected(case, rank)
    ref_out = compute_ref(**inputs, group=group)
    res_out = compute_res(**inputs, group=group)
    assert res_out.shape == want.shape == (want.shape[0], 1, case[3] + case[4])
    assert res_out.dtype == want.dtype == case[6]
    assert torch.equal(res_out, want), "candidate packed rows are wrong"
    assert torch.equal(ref_out, want), "reference packed rows are wrong"
    return dcp.time_pair(lambda: compute_ref(**inputs, group=group),
                         lambda: compute_res(**inputs, group=group))


def worker(rank, payload, rendezvous, outdir):
    dcp.dcp_worker(rank, payload, rendezvous, outdir, body)


@label("v1_sgl_dcp_gather_cache_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_dcp_gather_cache_(config):
    case, = config
    times = dcp.run(worker, {"world": case[0], "case": case})
    return timings(max(t[0] for t in times), max(t[1] for t in times))
