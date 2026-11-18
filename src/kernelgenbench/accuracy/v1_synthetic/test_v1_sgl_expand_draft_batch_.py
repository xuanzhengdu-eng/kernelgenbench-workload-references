"""Accuracy and performance for v1_sgl_expand_draft_batch_.

Contract (SGLang pinned frozen-KV MTP): expand one committed-prefix batch to
``B * topk`` draft frontiers.  Request-pool indices, sequence lengths, host
sequence lengths and positions are repeated per branch; the global token count
is filled in place; mrope positions are repeated only when their width matches
the expanded token count.  Returns None.  ``topk == 1`` and an empty batch are
no-ops, and an already-expanded batch (``batch_size != len(seq_lens)``) raises.
"""
import types

import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.spec_decode_test_utils import fresh, measure, to_device
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (batch size, topk, host seq_lens present, global scalar present,
#  mrope layout: valid / mismatch / none)
CASES = [
    (2, 2, True, True, "valid"),
    (3, 3, False, False, "mismatch"),
    (2, 2, False, True, "none"),
    (2, 1, True, True, "valid"),
    (0, 3, True, True, "valid"),
]
COMBOS = CASES

FIELDS = ("batch_size", "req_pool_indices", "seq_lens", "seq_lens_cpu", "seq_lens_sum",
          "positions", "global_num_token_non_padded_cpu", "global_num_token_non_padded",
          "mrope_positions")
TENSOR_FIELDS = ("req_pool_indices", "seq_lens", "seq_lens_cpu", "positions",
                 "global_num_token_non_padded", "mrope_positions")


def input_build(config):
    batch_size, topk, host_cpu, global_scalar, rope = config
    seq_lens = torch.arange(batch_size, dtype=torch.int32) * 3
    return {
        "batch": types.SimpleNamespace(
            batch_size=batch_size,
            req_pool_indices=torch.arange(batch_size, dtype=torch.int64) + 10,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.clone() if host_cpu else None,
            seq_lens_sum=123,  # stale value the entry must recompute
            positions=torch.full((batch_size,), -7, dtype=torch.int64),
            global_num_token_non_padded_cpu=99,  # stale value the entry must recompute
            global_num_token_non_padded=(torch.tensor([-9], dtype=torch.int32)
                                         if global_scalar else None),
            mrope_positions=(torch.arange(3 * batch_size, dtype=torch.int64).reshape(3, batch_size)
                             if rope == "valid"
                             else torch.ones((3, batch_size + 1), dtype=torch.int64)
                             if rope == "mismatch" else None),
        ),
        "topk": topk,
    }


def move(inputs, device):
    batch = inputs["batch"]
    moved = types.SimpleNamespace(**vars(batch))
    for field in TENSOR_FIELDS:
        value = getattr(batch, field)
        # seq_lens_cpu is host metadata by contract and stays on the CPU.
        if value is not None and field != "seq_lens_cpu":
            setattr(moved, field, to_device(value, device))
    return {"batch": moved, "topk": inputs["topk"]}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_expand_draft_batch_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_expand_draft_batch_(**inputs)


def compare(ref_batch, res_batch):
    for field in FIELDS:
        expected, actual = getattr(ref_batch, field), getattr(res_batch, field)
        if isinstance(expected, torch.Tensor):
            assert isinstance(actual, torch.Tensor), field
            assert actual.dtype == expected.dtype and actual.shape == expected.shape, field
            assert torch.equal(actual, expected), f"{field} differs from reference"
        else:
            assert actual == expected, f"{field} differs from reference"


def expected_batch(config):
    """Independent expansion of the batch fields, or None for a no-op config."""
    batch_size, topk, host_cpu, global_scalar, rope = config
    if topk == 1 or batch_size == 0:
        return None
    seq_lens = (torch.arange(batch_size, dtype=torch.int32) * 3).repeat_interleave(topk, dim=0)
    positions = (seq_lens - 1).clamp_min(0).to(torch.int64)
    mrope = None
    if rope == "valid":
        mrope = torch.arange(3 * batch_size, dtype=torch.int64).reshape(3, batch_size)
        if mrope.shape[-1] * topk == positions.numel():
            mrope = mrope.repeat_interleave(topk, dim=-1)
    elif rope == "mismatch":
        mrope = torch.ones((3, batch_size + 1), dtype=torch.int64)
    return {
        "batch_size": batch_size * topk,
        "req_pool_indices": (torch.arange(batch_size, dtype=torch.int64) + 10).repeat_interleave(topk, dim=0),
        "seq_lens": seq_lens,
        "seq_lens_cpu": seq_lens.clone() if host_cpu else None,
        "seq_lens_sum": int(seq_lens.sum()),
        "positions": positions,
        "global_num_token_non_padded_cpu": positions.numel(),
        "mrope_positions": mrope,
    }


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs, device=kernelgenbench.device)


@label("v1_sgl_expand_draft_batch_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_expand_draft_batch_(config):
    inputs = move(input_build(config), kernelgenbench.device)
    ref_inputs = move(input_build(config), kernelgenbench.device)
    res_inputs = move(input_build(config), kernelgenbench.device)
    expected = expected_batch(config)

    ref_scalar = ref_inputs["batch"].global_num_token_non_padded
    res_scalar = res_inputs["batch"].global_num_token_non_padded
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    assert ref_out is None and res_out is None
    compare(ref_inputs["batch"], res_inputs["batch"])

    # The batch is expanded in place: batch_size and every repeated field follow
    # B * topk, while the global token count is a fill_ (same object).
    ref_batch = ref_inputs["batch"]
    if expected is None:
        assert ref_batch.batch_size == config[0]
        assert ref_batch.seq_lens_sum == 123
        assert ref_batch.global_num_token_non_padded_cpu == 99
        assert torch.equal(ref_batch.positions, torch.full((config[0],), -7, dtype=torch.int64,
                                                           device=ref_batch.positions.device))
        assert torch.equal(ref_batch.seq_lens, torch.arange(config[0], dtype=torch.int32,
                                                            device=ref_batch.seq_lens.device) * 3)
    else:
        assert ref_batch.batch_size == expected["batch_size"]
        assert torch.equal(ref_batch.req_pool_indices, expected["req_pool_indices"].to(ref_batch.req_pool_indices.device))
        assert torch.equal(ref_batch.seq_lens, expected["seq_lens"].to(ref_batch.seq_lens.device))
        assert torch.equal(ref_batch.positions, expected["positions"].to(ref_batch.positions.device))
        assert ref_batch.positions.dtype == torch.int64
        assert ref_batch.seq_lens_sum == expected["seq_lens_sum"]
        assert ref_batch.global_num_token_non_padded_cpu == expected["global_num_token_non_padded_cpu"]
        if expected["seq_lens_cpu"] is not None:
            assert torch.equal(ref_batch.seq_lens_cpu, expected["seq_lens_cpu"])
        if expected["mrope_positions"] is not None:
            assert torch.equal(ref_batch.mrope_positions, expected["mrope_positions"].to(ref_batch.mrope_positions.device)), "mrope layout not repeated correctly"
    if ref_scalar is not None:
        assert ref_batch.global_num_token_non_padded is ref_scalar
        if expected is not None:
            assert int(ref_batch.global_num_token_non_padded.item()) == expected["global_num_token_non_padded_cpu"]
        else:
            assert int(ref_batch.global_num_token_non_padded.item()) == -9
        assert res_inputs["batch"].global_num_token_non_padded is res_scalar

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
