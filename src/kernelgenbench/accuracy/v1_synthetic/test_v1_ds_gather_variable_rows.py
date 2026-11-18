"""DeepSpeed padded all-gather of unequal row counts.

``counts`` holds every peer's row count and ``max_rows`` its maximum.  Each rank
pads to ``max_rows``, the group all-gathers the equal-shaped buffer, and the result
is the rank-ordered concatenation trimmed to ``counts``.  ``(0,)``/``(0, 0)``
counts produce a fresh empty tensor without any payload.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.dist_collective_harness import aggregate, launch, timed
from sandbox.verifier.test_parametrize import label, parametrize

# world size, counts per rank
COMBOS = [
    (1, (3,)),
    (1, (0,)),
    (2, (2, 3)),
    (2, (0, 3)),
    (2, (0, 0)),
    (2, (3, 3)),
]


def input_build(config, rank, dtype=torch.float32):
    world, counts = config
    rows = counts[rank]
    return {"input_": torch.arange(rows * 3, dtype=dtype).reshape(rows, 3) + rank * 10,
            "counts": counts, "max_rows": max(counts)}


def compute_ref(input_, counts, max_rows, group):
    return kernelgenbench.baseline.v1_ds_gather_variable_rows(input_, group, counts, max_rows)


def compute_res(input_, counts, max_rows, group):
    return kernelgenbench.solution.v1_ds_gather_variable_rows(input_, group, counts, max_rows)


def compare(config, rank, ref_out, res_out, original, dtype):
    world, counts = config
    expected = torch.cat([input_build(config, peer, dtype)["input_"] for peer in range(world)], dim=0)
    assert expected.shape[0] == sum(counts)
    for out in (ref_out, res_out):
        assert isinstance(out, torch.Tensor)
        assert out.shape == expected.shape and out.dtype == dtype
        assert out.is_contiguous()
        assert torch.equal(out, expected)
    if max(counts) == 0:
        assert ref_out.numel() == res_out.numel() == 0
    else:
        assert ref_out.data_ptr() != original.data_ptr()
        assert res_out.data_ptr() != original.data_ptr()


def _worker(rank, config, dtype):
    payload = input_build(config, rank, dtype)
    group = torch.distributed.group.WORLD
    before = payload["input_"].clone()
    ref_out = compute_ref(**payload, group=group)
    res_out = compute_res(**payload, group=group)
    compare(config, rank, ref_out, res_out, payload["input_"], dtype)
    assert torch.equal(payload["input_"], before)
    return {"ref_ms": timed(lambda: compute_ref(**payload, group=group)),
            "res_ms": timed(lambda: compute_res(**payload, group=group))}


@label("v1_ds_gather_variable_rows")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("config,dtype", [(config, dtype) for config in COMBOS
                                          for dtype in (torch.float32, torch.int64)])
def test_accuracy_v1_ds_gather_variable_rows(config, dtype):
    return aggregate(launch(_worker, config[0], (config, dtype)))
