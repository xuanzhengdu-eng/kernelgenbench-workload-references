"""Official vLLM row-parallel linear with real Gloo TP collectives.

The weight is sharded per rank exactly as the real layer's partition is. The
test checks the reduced output, bias placement (fused on rank 0 before the
all-reduce, or returned when deferred), the local partial product when
`reduce_results=False`, and that inputs are untouched; both sides are timed.
"""
import datetime
import json
import os
import tempfile
import pickle
from types import SimpleNamespace
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from kernelgenbench.dataset.baseline.v1_synthetic.v1_vllm_row_parallel_linear import v1_vllm_row_parallel_linear

# (world, input already sharded, deferred bias, reduce results, return bias,
#  dtype, has bias)
COMBOS = [
    (1, False, False, True, True, torch.float32, True),
    (2, False, False, True, True, torch.float32, True),
    (2, True, True, True, True, torch.float32, True),
    (2, True, False, False, True, torch.bfloat16, False),
    (2, True, False, True, False, torch.float32, True),
    (2, True, False, True, True, torch.float32, False),
]


def get_input(case, rank, device='cuda'):
    world, parallel, skip, reduce_results, ret_bias, dtype, has_bias = case
    gen = torch.Generator(device=device).manual_seed(52)
    full = torch.randn((4, 2, world * 8), generator=gen, dtype=dtype, device=device)
    weight = torch.randn((6, 8), generator=torch.Generator(device=device).manual_seed(100 + rank),
                         dtype=dtype, device=device)
    bias = (torch.randn((6,), generator=torch.Generator(device=device).manual_seed(11),
                        dtype=dtype, device=device) if has_bias else None)
    x = full.chunk(world, -1)[rank].contiguous() if parallel else full
    return dict(x=x, weight=weight, bias=bias, input_is_parallel=parallel,
                skip_bias_add=skip, reduce_results=reduce_results, return_bias=ret_bias)


def input_build(case, rank):
    return get_input(case, rank)


def compute_ref(**d):
    return v1_vllm_row_parallel_linear(**d)


def compute_res(**d):
    import kernelgenbench
    return kernelgenbench.triton.v1_vllm_row_parallel_linear(**d)


def worker(rank, case, path, outdir, candidate_bytes):
    import kernelgenbench
    kernelgenbench.triton = SimpleNamespace(v1_vllm_row_parallel_linear=pickle.loads(candidate_bytes))
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank,
                            world_size=case[0], timeout=datetime.timedelta(seconds=60))
    try:
        a = get_input(case, rank)
        b = get_input(case, rank)
        world, parallel, skip, reduce_results, ret_bias, dtype, has_bias = case
        expected = compute_ref(**a, group=dist.group.WORLD)
        actual = compute_res(**b, group=dist.group.WORLD)
        if not ret_bias:
            assert isinstance(actual, torch.Tensor)
            torch.testing.assert_close(expected, actual, rtol=.02, atol=.02)
        else:
            assert isinstance(actual, tuple) and len(actual) == 2
            torch.testing.assert_close(expected[0], actual[0], rtol=.02, atol=.02)
            assert (expected[1] is None) == (actual[1] is None)
            if actual[1] is not None:
                # Upstream returns the module-owned bias parameter, so value
                # equality is the contract here (not object identity with the
                # caller's bias tensor).
                assert actual[1].shape == b['bias'].shape
                torch.testing.assert_close(expected[1], actual[1], rtol=0, atol=0)
        for key in ('x', 'weight', 'bias'):
            if a[key] is not None:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

        def bench(fn):
            dist.barrier()
            start = time.perf_counter()
            for _ in range(3):
                fn(**get_input(case, rank), group=dist.group.WORLD)
            dist.barrier()
            return (time.perf_counter() - start) * 1000 / 3

        ta, tb = bench(compute_ref), bench(compute_res)
        with open(os.path.join(outdir, str(rank) + '.json'), 'w') as f:
            json.dump([ta, tb], f)
    finally:
        dist.destroy_process_group()


@label('v1_vllm_row_parallel_linear')
@parametrize('case', COMBOS)
@pytest.mark.parametrize('case', COMBOS)
def test_accuracy_v1_vllm_row_parallel_linear(case):
    import kernelgenbench
    candidate = kernelgenbench.triton.v1_vllm_row_parallel_linear
    try:
        candidate_bytes = pickle.dumps(candidate)
    except (pickle.PicklingError, AttributeError, TypeError):
        import cloudpickle
        candidate_bytes = cloudpickle.dumps(candidate)
    with tempfile.TemporaryDirectory(prefix='v1-row-parallel-vllm-') as d:
        mp.start_processes(worker, args=(case, os.path.join(d, 'rdzv'), d, candidate_bytes),
                           nprocs=case[0], join=True, start_method='spawn')
        times = [json.load(open(os.path.join(d, str(r) + '.json'))) for r in range(case[0])]
    ta = max(t[0] for t in times)
    tb = max(t[1] for t in times)
    return CustomBenchmarkResult(ref_time=ta, res_time=tb, speedup=ta / tb if tb else float('inf'))
