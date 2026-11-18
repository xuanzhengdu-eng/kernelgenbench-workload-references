"""Check paged decode against the pinned vLLM TritonMLAImpl.forward_mqa on cuda."""
import time
import pytest
import torch
import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult

COMBOS = [(torch.float16, 1, (1, 3), False), (torch.float16, 4, (17, 31), True), (torch.bfloat16, 8, (5, 20), False), (torch.bfloat16, 2, (16, 32), True), (torch.float32, 2, (9, 24), False)]


def input_build(config):
    dtype, heads, lengths, split_query = config
    gen = torch.Generator().manual_seed(1960 + heads + sum(lengths))
    query = (torch.randn((len(lengths), heads, 576), generator=gen) * .15).to(dtype=dtype, device="cuda")
    cache = (torch.randn((5, 16, 576), generator=gen) * .2).to(dtype=dtype, device="cuda")
    if split_query:
        query = query.split((512, 64), dim=-1)
    data = dict(query=query, kv_cache=cache, scale=576 ** -.5,
                block_tables=torch.tensor([[2, 0, 1], [4, 1, 0]], dtype=torch.int32, device="cuda"),
                seq_lens=torch.tensor(lengths, dtype=torch.int32, device="cuda"))
    
    return data


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_mla_decode_triton(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_mla_decode_triton(**inputs)


def elapsed(fn, inputs):
    if inputs["kv_cache"].is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        fn(**inputs)
    if inputs["kv_cache"].is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 200


@label("v1_vllm_mla_decode_triton")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_mla_decode_triton(config):
    reference_inputs, solution_inputs = input_build(config), input_build(config)
    saved = {k: v.clone() for k, v in solution_inputs.items() if torch.is_tensor(v) and k != "out"}
    query = solution_inputs["query"]
    query_tensors = query if isinstance(query, tuple) else (query,)
    saved_query = [t.clone() for t in query_tensors]
    expected = compute_ref(**reference_inputs)
    actual = compute_res(**solution_inputs)
    assert isinstance(actual, tuple) and len(actual) == 2
    for observed, wanted in zip(actual, expected):
        assert observed.device == wanted.device and observed.dtype == config[0]
        assert observed.shape == wanted.shape
        assert not torch._C._is_alias_of(observed, solution_inputs["kv_cache"])
        torch.testing.assert_close(observed, wanted, atol=.012 if config[0]==torch.bfloat16 else .002, rtol=.008 if config[0]==torch.bfloat16 else .001)
    for key, before in saved.items():
        assert torch.equal(solution_inputs[key], before)
    for t, before in zip(query_tensors, saved_query):
        assert torch.equal(t, before)
    t_ref = elapsed(compute_ref, reference_inputs)
    t_res = elapsed(compute_res, solution_inputs)
    return CustomBenchmarkResult(ref_time=t_ref, res_time=t_res, speedup=t_ref / t_res)
