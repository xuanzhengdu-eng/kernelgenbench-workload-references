"""Accuracy and performance for v1_cp_position_embedding (Megatron CP selection).

Official source (NVIDIA/Megatron-LM pinned commit 266f1c97):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/rope_utils.py#L48-L70

Contract: ``pos_emb`` is ``[B, T, H]`` with ``seq_dim=1`` and T divisible by
``2 * cp_size``.  The sequence axis is viewed as ``[2*cp_size, T/(2*cp_size)]``;
the chunk at ``cp_rank`` and the mirrored chunk at ``2*cp_size - cp_rank - 1``
are selected and concatenated, giving ``[B, 2*T/(2*cp_size), H]``.  The CP
process group is represented by the scalar ``cp_size``/``cp_rank`` because the
selection performs no communication; the verification binding passes a real
rank/size pair through the official ``cp_group`` argument.  The result is a
fresh tensor; the input is untouched.

Item: v1_cp_position_embedding (723-row 118).  The old reference cited the
flagos-ai Megatron-LM-FL fork; the official pinned Megatron has the same
function, so the reference now points at the official file and the quarantine
of this test in ``v1_synthetic/__init__.py`` is lifted.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows112_121_checks as checks


# (shape (B, T, H), cp_size, cp_rank, dtype): T is divisible by 2*cp_size.
CASES = [
    ((2, 16, 128), 2, 0, torch.float32),
    ((2, 16, 128), 2, 1, torch.float16),
    ((2, 128, 1024), 2, 0, torch.bfloat16),
    ((2, 128, 1024), 2, 1, torch.float32),
    ((3, 32, 64), 4, 0, torch.bfloat16),
    ((3, 32, 64), 4, 1, torch.float16),
    ((3, 32, 64), 4, 2, torch.float32),
    ((3, 32, 64), 4, 3, torch.bfloat16),
]

# Axes: one combo per case.
COMBOS = list(CASES)


def input_build(config):
    """pos_emb [B, T, H] on CPU with a fixed seed; the two scalars select the rank."""
    shape, cp_size, cp_rank, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261118)
    pos_emb = torch.randn(shape, dtype=dtype, generator=generator)
    return dict(pos_emb=pos_emb, cp_size=cp_size, cp_rank=cp_rank)


def compute_ref(pos_emb, cp_size, cp_rank):
    return kernelgenbench.baseline.v1_cp_position_embedding(pos_emb, cp_size, cp_rank)


def compute_res(pos_emb, cp_size, cp_rank):
    return kernelgenbench.solution.v1_cp_position_embedding(pos_emb, cp_size, cp_rank)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor), f"expected a tensor, got {type(res_out).__name__}"
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == ref_out.device
    # Pure selection: bit-identical values, no arithmetic.
    assert torch.equal(res_out, ref_out), "selection differs from the official mapping"


def compute_ref_performance(pos_emb, cp_size, cp_rank):
    return checks.measure(kernelgenbench.baseline.v1_cp_position_embedding,
                          {"pos_emb": pos_emb, "cp_size": cp_size, "cp_rank": cp_rank},
                          kernelgenbench.device)


def compute_res_performance(pos_emb, cp_size, cp_rank):
    return checks.measure(kernelgenbench.solution.v1_cp_position_embedding,
                          {"pos_emb": pos_emb, "cp_size": cp_size, "cp_rank": cp_rank},
                          kernelgenbench.device)


@label("v1_cp_position_embedding")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_cp_position_embedding(config):
    shape, cp_size, cp_rank, dtype = config
    assert shape[1] % (2 * cp_size) == 0
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    assert res_inputs["pos_emb"].shape == shape

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # The mirrored pair is what distinguishes the mapping from a naive left slice.
    chunk = shape[1] // (2 * cp_size)
    assert ref_out.shape == (shape[0], 2 * chunk, shape[2])
    expected = torch.cat((inputs["pos_emb"][:, cp_rank * chunk:(cp_rank + 1) * chunk],
                          inputs["pos_emb"][:, (2 * cp_size - cp_rank - 1) * chunk:
                                            (2 * cp_size - cp_rank) * chunk]), dim=1)
    assert torch.equal(ref_out.cpu(), expected), "reference mapping is not the mirrored pair"
    assert not torch._C._is_alias_of(ref_out, res_inputs["pos_emb"])
    assert torch.equal(res_inputs["pos_emb"], ref_inputs["pos_emb"]), "input was modified"

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
