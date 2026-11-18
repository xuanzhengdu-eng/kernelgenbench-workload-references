"""Accuracy and performance for v1_sgl_tree_target_only_.

Tree target-only verification walks a fixed retrieval tree: from the root, for
each verify level it scans the children (``retrive_next_token``) and then their
siblings (``retrive_next_sibling``), accumulating target mass until a coin
accepts a draft token (``coin < acc / threshold_acc`` or ``p >= threshold_single``).
On rejection the bonus token is drawn from ``relu(target - draft)`` at the visited
node; when the whole draft is accepted, from the pure target row.  All outputs are
caller buffers written in place and the entry returns ``None``.

Key assertions: ``accept_token_num`` per batch, ``accept_index`` over the accepted
prefix (root included), ``predicts`` at those indices, the ``draft_probs`` values
recorded for rejected siblings, unchanged inputs, and preserved buffer identity.

Reference: Torch implementation of the fixed-commit tree verify.
Solution: SGLang's JIT ``tree_speculative_sampling_target_only`` kernel.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.row_pair_utils import fresh, measure_cuda, to_device
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


CASES = [
    # batch, accept-index width, threshold_single, threshold_acc
    (1, 1, 1.0, 1.0),
    (2, 4, 1.0, 1.0),
    (3, 3, 0.15, 1.0),
    (2, 5, 0.8, 0.35),
    (4, 2, 0.0, 1.0),
]
COMBOS = CASES
NODES = 6
VOCAB = 11


def input_build(config):
    batch, spec, threshold_single, threshold_acc = config
    generator = torch.Generator(device="cpu").manual_seed(562 + batch * 100 + spec)
    tree = torch.tensor([1, 3, 5, -1, -1, -1], dtype=torch.int64).repeat(batch, 1)
    sibling = torch.tensor([-1, 2, -1, 4, -1, -1], dtype=torch.int64).repeat(batch, 1)
    return {
        "predicts": torch.full((batch * NODES,), -99, dtype=torch.int32),
        "accept_index": torch.full((batch, spec), -99, dtype=torch.int32),
        "accept_token_num": torch.full((batch,), -99, dtype=torch.int32),
        "candidates": torch.randint(VOCAB, (batch, NODES), generator=generator,
                                    dtype=torch.int64),
        "retrive_index": torch.arange(batch * NODES, dtype=torch.int64).view(batch, NODES),
        "retrive_next_token": tree,
        "retrive_next_sibling": sibling,
        "uniform_samples": torch.rand(batch, NODES, generator=generator),
        "uniform_samples_for_final_sampling": torch.rand(batch, generator=generator),
        "target_probs": torch.softmax(torch.randn(batch, NODES, VOCAB, generator=generator), -1),
        "draft_probs": torch.zeros(batch, NODES, VOCAB),
        "threshold_single": threshold_single,
        "threshold_acc": threshold_acc,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_tree_target_only_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_tree_target_only_(**inputs)


MUTABLE = ("predicts", "accept_index", "accept_token_num", "draft_probs")


def compare(ref_out, res_out, ref_inputs, res_inputs):
    assert ref_out is None and res_out is None
    assert torch.equal(res_inputs["accept_token_num"], ref_inputs["accept_token_num"])
    assert torch.equal(res_inputs["draft_probs"], ref_inputs["draft_probs"])
    for row in range(ref_inputs["accept_token_num"].shape[0]):
        count = int(ref_inputs["accept_token_num"][row]) + 1
        indices = ref_inputs["accept_index"][row, :count]
        assert torch.equal(indices, res_inputs["accept_index"][row, :count])
        assert torch.equal(ref_inputs["predicts"][indices.long()],
                           res_inputs["predicts"][indices.long()])


def compute_ref_performance(inputs):
    return measure_cuda(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure_cuda(compute_res, inputs)


@label("v1_sgl_tree_target_only_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_tree_target_only_(config):
    batch, spec, _, _ = config
    inputs = to_device(input_build(config))
    ref_inputs, res_inputs = fresh(inputs), fresh(inputs)
    assert ref_inputs["candidates"].shape == (batch, NODES)
    assert ref_inputs["target_probs"].shape == (batch, NODES, VOCAB)
    pointers = {name: res_inputs[name].data_ptr() for name in MUTABLE}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs)

    for name, pointer in pointers.items():
        assert res_inputs[name].data_ptr() == pointer, f"{name} was replaced"
    for name in ("candidates", "retrive_index", "retrive_next_token",
                 "retrive_next_sibling", "uniform_samples", "target_probs",
                 "uniform_samples_for_final_sampling"):
        torch.testing.assert_close(res_inputs[name], inputs[name], rtol=0, atol=0)

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
