"""Accuracy and performance for v1_ds_partition_grads_.

Operator: deepspeed.runtime.zenflow.engine_stage3.partition_grads (native torch).
Official source (pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/zenflow/engine_stage3.py#L457-L527

Item: v1_ds_partition_grads_ (723-row 164). ``compute_ref`` loads this file's
reference from ``kernelgenbench.baseline``; ``compute_res`` calls the real official
``partition_grads`` bound as ``kernelgenbench.solution`` with a one-rank NCCL
group, so ``dist.get_rank`` returns 0.

Inputs are three registered-style parameters (``partition_numel``, ``ds_numel``,
``ds_id``, a live ``grad``), their gradient partitions, and persistent fp32 grad
buffers. Cases cover micro step 0 (overwrite), a later micro step (accumulate in
fp32), an empty partition (``ds_numel == 0``, skipped) and a bf16 partition cast
into the fp32 buffer. Key assertions: the returned list aliases the persistent
buffers in parameter order, values match overwrite/accumulate semantics, skipped
parameters contribute no buffer, every visited ``param.grad`` is released, and the
incoming partitions are untouched.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows162_171_checks as checks


# (rank, micro step, partition lengths, param ds_numels, partition dtype)
COMBOS = [
    (0, 0, (4, 4, 4), (6, 4, 8), torch.float32),
    (0, 2, (4, 4, 4), (6, 4, 8), torch.float32),
    (0, 0, (0, 0, 0), (0, 0, 0), torch.float32),
    (0, 3, (2, 0, 4), (6, 4, 8), torch.bfloat16),
]

PARTITION_NUMEL = 4
DS_IDS = (7, 8, 9)


def input_build(config):
    rank, micro_step, lengths, ds_numels, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    partitions = [torch.randn((n,), generator=generator, dtype=dtype) for n in lengths]
    buffers = {ds_id: torch.randn((4,), generator=generator, dtype=torch.float32)
               for ds_id in DS_IDS}
    meta = [(ds_id, PARTITION_NUMEL, ds_numel)
            for ds_id, ds_numel in zip(DS_IDS, ds_numels)]
    return {
        "param_meta": meta,
        "grad_partitions": partitions,
        "grad_buffers": buffers,
        "rank": rank,
        "micro_step": micro_step,
        "accum_dtype": torch.float32,
        "param_dtype": dtype,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_partition_grads_(
        params=inputs["params"], grad_partitions=inputs["grad_partitions"],
        grad_buffers=inputs["grad_buffers"], rank=inputs["rank"],
        micro_step=inputs["micro_step"], accum_dtype=inputs["accum_dtype"])


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_partition_grads_(
        params=inputs["params"], grad_partitions=inputs["grad_partitions"],
        grad_buffers=inputs["grad_buffers"], rank=inputs["rank"],
        micro_step=inputs["micro_step"], accum_dtype=inputs["accum_dtype"])


def _copy(inputs):
    device = kernelgenbench.device
    params = []
    for ds_id, partition_numel, ds_numel in inputs["param_meta"]:
        param = torch.nn.Parameter(torch.ones(1, device=device))
        param.partition_numel = lambda value=partition_numel: value
        param.ds_numel = ds_numel
        param.ds_id = ds_id
        param.grad = torch.ones(1, device=device)
        params.append(param)
    return {
        "param_meta": inputs["param_meta"],
        "params": params,
        "grad_partitions": [t.clone().to(device) for t in inputs["grad_partitions"]],
        "grad_buffers": {k: v.clone().to(device) for k, v in inputs["grad_buffers"].items()},
        "rank": inputs["rank"],
        "micro_step": inputs["micro_step"],
        "accum_dtype": inputs["accum_dtype"],
    }


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, list) and isinstance(ref_out, list)
    assert len(ref_out) == len(res_out)
    for expected, actual in zip(ref_out, res_out):
        assert expected.shape == actual.shape and expected.dtype == actual.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _check_side(output, inputs, config, before):
    _, micro_step, lengths, ds_numels, dtype = config
    buffers = inputs["grad_buffers"]
    expected_values = {ds_id: value.clone() for ds_id, value in before.items()}
    visited = 0
    for index, (ds_id, partition_numel, ds_numel) in enumerate(inputs["param_meta"]):
        if partition_numel * inputs["rank"] >= ds_numel:
            assert inputs["params"][index].grad is None
            continue
        visited += 1
        target = output[visited - 1]
        assert torch._C._is_alias_of(target, buffers[ds_id]), \
            "returned buffer must alias the persistent grad buffer"
        partition = inputs["grad_partitions"][index].float()
        if micro_step == 0:
            expected_values[ds_id][:partition.numel()] = partition
        else:
            expected_values[ds_id][:partition.numel()] += partition
        assert inputs["params"][index].grad is None
    assert visited == len(output)
    for ds_id, expected in expected_values.items():
        torch.testing.assert_close(buffers[ds_id], expected, rtol=0, atol=0)
        assert buffers[ds_id].dtype == torch.float32


def compute_ref_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_ref,
                                  kernelgenbench.device)


def compute_res_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_res,
                                  kernelgenbench.device)


@label("v1_ds_partition_grads_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_partition_grads_(config):
    inputs = input_build(config)
    ref_inputs = _copy(inputs)
    res_inputs = _copy(inputs)
    ref_partitions = [t.clone() for t in ref_inputs["grad_partitions"]]
    res_partitions = [t.clone() for t in res_inputs["grad_partitions"]]

    ref_before = {k: v.float().clone() for k, v in ref_inputs["grad_buffers"].items()}
    res_before = {k: v.float().clone() for k, v in res_inputs["grad_buffers"].items()}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)
    _check_side(ref_out, ref_inputs, config, ref_before)
    _check_side(res_out, res_inputs, config, res_before)

    for expected, actual in zip(ref_partitions, ref_inputs["grad_partitions"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for expected, actual in zip(res_partitions, res_inputs["grad_partitions"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return checks.timings(ref_time, res_time)
