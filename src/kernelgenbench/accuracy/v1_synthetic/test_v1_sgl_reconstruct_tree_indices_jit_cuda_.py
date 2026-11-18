# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/jit/csrc/speculative/tree.cuh#L171-L208
"""Verify in-place tree reconstruction using independent per-node expectations."""
import time
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

TREES = [((-1,),), ((-1, 0, 0, 1, 1, 2), (-1, 0, 1, 2, 3, 4)), ((-1, 0, 0, 0),)]
COMBOS = [(trees, dtype) for trees in TREES for dtype in (torch.bool, torch.uint8)]


def input_build(config):
    trees, dtype = config
    batch, n = len(trees), len(trees[0])
    device = "cuda"
    lengths = torch.arange(7, 7 + batch, device=device, dtype=torch.int64)
    mask = torch.eye(n, device=device, dtype=dtype).expand(batch, n, n).clone()
    for row, parents in enumerate(trees):
        for node, parent in enumerate(parents):
            while parent >= 0:
                mask[row, node, parent] = True
                parent = parents[parent]
    links = [torch.full((batch, n), -99, device=device, dtype=lengths.dtype) for _ in range(3)]
    return dict(tree_mask=torch.cat((mask.flatten(), mask.new_zeros(3))), verified_seq_len=lengths, positions=lengths.new_full((batch*n,), -99),
                retrive_index=links[0], retrive_next_token=links[1], retrive_next_sibling=links[2],
                batch_size=batch, draft_token_num=n)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_reconstruct_tree_indices_jit_cuda_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_reconstruct_tree_indices_jit_cuda_(**inputs)


def compare(out, inputs, trees):
    assert out is None
    n = len(trees[0])
    for b, parents in enumerate(trees):
        for i, parent in enumerate(parents):
            depth, ancestor = 0, parent
            while ancestor >= 0:
                depth += 1
                ancestor = parents[ancestor]
            child = next((j for j in range(i+1, n) if parents[j] == i), -1)
            sibling = next((j for j in range(i+1, n) if parent >= 0 and parents[j] == parent), -1)
            assert int(inputs["positions"].view(-1)[b*n+i]) == 7 + b + depth
            assert int(inputs["retrive_index"][b, i]) == b*n + i
            assert int(inputs["retrive_next_token"][b, i]) == child
            assert int(inputs["retrive_next_sibling"][b, i]) == sibling


@label("v1_sgl_reconstruct_tree_indices_jit_cuda_")
@parametrize("config", COMBOS)
def test_accuracy_v1_sgl_reconstruct_tree_indices_jit_cuda_(config):
    times = []
    for fn in (compute_ref, compute_res):
        inputs = input_build(config)
        saved = {key: inputs[key].clone() for key in ("tree_mask", "verified_seq_len")}
        ptrs = {key: obj.data_ptr() for key, obj in inputs.items() if isinstance(obj, torch.Tensor)}
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn(**inputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        compare(out, inputs, config[0])
        assert all(inputs[key].data_ptr() == value for key, value in ptrs.items())
        for key, value in saved.items():
            torch.testing.assert_close(inputs[key], value, atol=0, rtol=0)
    return CustomBenchmarkResult(ref_time=times[0], res_time=times[1], speedup=times[0] / times[1])

