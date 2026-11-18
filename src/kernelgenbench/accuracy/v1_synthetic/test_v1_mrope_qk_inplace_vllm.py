# Synthetic workload; no production capture or production frequency claim.
"""OP-540d262db7f5 (vLLM branch): MRotaryEmbedding.forward_cuda Triton kernel.

positions is (3, tokens) T/H/W int64; query/key are (tokens, heads*head_size);
cos_sin_cache is (max_position, rotary_dim) with cosine then sine halves.  The
vLLM Triton kernel mutates Q/K in place and returns reshaped views of them.
"""
import statistics
import torch
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import kernelgenbench


COMBOS = [
    (dtype, interleaved, neox, tokens, rotary_dim)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for (interleaved, neox, tokens, rotary_dim) in (
        (False, True, 16, 32),
        (False, False, 5, 32),
        (True, True, 9, 64),
        # vLLM pads the rotary width, so an interleaved partial width works.
        (True, True, 7, 32),
    )
]


def input_build(config):
    dtype, interleaved, neox, tokens, rotary_dim = config
    gen = torch.Generator(device="cpu").manual_seed(138)
    head_size = 64
    seq = torch.arange(tokens)
    positions = torch.stack([seq % 7, seq % 5, seq % 3], dim=0).to(torch.int64)
    angle = torch.randn((48, rotary_dim // 2), generator=gen, dtype=torch.float32)
    cache = torch.cat([angle.cos(), angle.sin()], dim=-1).to(dtype)
    q = torch.randn((tokens, 4 * head_size), generator=gen, dtype=torch.float32).to(dtype)
    k = torch.randn((tokens, 2 * head_size), generator=gen, dtype=torch.float32).to(dtype)
    return {"positions": positions.to(kernelgenbench.device),
            "query": q.to(kernelgenbench.device), "key": k.to(kernelgenbench.device),
            "cos_sin_cache": cache.to(kernelgenbench.device),
            "head_size": head_size, "rotary_dim": rotary_dim,
            "mrope_section": [rotary_dim // 4, rotary_dim // 8, rotary_dim // 2 - rotary_dim // 4 - rotary_dim // 8],
            "mrope_interleaved": interleaved, "is_neox_style": neox}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_mrope_qk_inplace_vllm(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_mrope_qk_inplace_vllm(**inputs)


def compare(ref_out, res_out, config):
    dtype = config[0]
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    rtol, atol = {torch.float32: (2e-5, 2e-5),
                  torch.float16: (3e-3, 3e-3),
                  torch.bfloat16: (3e-2, 3e-2)}[dtype]
    for expected, actual in zip(ref_out, res_out):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _time_inplace(fn, inputs, warmup=10, reps=50):
    def fresh():
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    for _ in range(warmup):
        fn(**fresh())
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        args = fresh()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(**args)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def compute_ref_performance(inputs):
    return _time_inplace(compute_ref, inputs)


def compute_res_performance(inputs):
    return _time_inplace(compute_res, inputs)


@label("v1_mrope_qk_inplace_vllm")
@parametrize("config", COMBOS)
def test_accuracy_v1_mrope_qk_inplace_vllm(config):
    dtype, interleaved, neox, tokens, rotary_dim = config
    inputs = input_build(config)
    assert inputs["positions"].shape == (3, tokens)
    assert inputs["query"].shape == (tokens, 256)
    assert inputs["key"].shape == (tokens, 128)

    original_q = inputs["query"].clone()
    ref_inputs = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    res_inputs = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    # vLLM returns reshape views of the mutated inputs.
    for out, inp in zip(res_out, (res_inputs["query"], res_inputs["key"])):
        assert out.data_ptr() == inp.data_ptr()
        assert torch.equal(out, inp)
    assert torch.equal(ref_out[0], ref_inputs["query"])
    assert not torch.equal(res_inputs["query"], original_q), "rotation must change Q"
    assert torch.equal(res_inputs["cos_sin_cache"].cpu(), inputs["cos_sin_cache"].cpu())

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
