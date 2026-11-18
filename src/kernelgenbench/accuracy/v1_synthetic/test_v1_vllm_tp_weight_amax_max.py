"""vLLM weight-amax MAX all-reduce: elementwise maximum over the TP peers.

Each rank holds one shard's ``amax``; the maximum over the group is the scale the
whole weight would produce.  The tensor is reduced in place and the same handle is
returned; the reduction is unconditional.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.dist_collective_harness import aggregate, launch, timed
from sandbox.verifier.test_parametrize import label, parametrize

# world size, amax length
COMBOS = [(world, length) for world in (1, 2) for length in (1, 4)]


def input_build(config, rank, dtype=torch.float32):
    world, length = config
    values = torch.arange(length, dtype=dtype) + rank * length + 1
    return {"amax": values}


def compute_ref(amax, group):
    return kernelgenbench.baseline.v1_vllm_tp_weight_amax_max(amax, group)


def compute_res(amax, group):
    return kernelgenbench.solution.v1_vllm_tp_weight_amax_max(amax, group)


def compare(config, rank, ref_out, res_out, ref_input, res_input, dtype):
    world, length = config
    expected = torch.stack([input_build(config, peer, dtype)["amax"] for peer in range(world)]).amax(0)
    assert ref_out is ref_input and res_out is res_input
    for tensor in (ref_input, res_input):
        assert torch.equal(tensor, expected), "MAX reduction differs from the group maximum"
        assert tensor.dtype == dtype and tensor.shape == (length,)


def _worker(rank, config, dtype):
    ref_payload = input_build(config, rank, dtype)
    res_payload = input_build(config, rank, dtype)
    group = torch.distributed.group.WORLD
    ref_out = compute_ref(**ref_payload, group=group)
    res_out = compute_res(**res_payload, group=group)
    compare(config, rank, ref_out, res_out, ref_payload["amax"], res_payload["amax"], dtype)
    return {"ref_ms": timed(lambda: compute_ref(**ref_payload, group=group)),
            "res_ms": timed(lambda: compute_res(**res_payload, group=group))}


@label("v1_vllm_tp_weight_amax_max")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("config,dtype", [(config, dtype) for config in COMBOS
                                          for dtype in (torch.float32, torch.int64)])
def test_accuracy_v1_vllm_tp_weight_amax_max(config, dtype):
    return aggregate(launch(_worker, config[0], (config, dtype)))
