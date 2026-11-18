# Synthetic workload; no production capture or production frequency claim.
"""OP-8b62b7b19686 (Triton branch): DeepSpeed fused RoPE backward.

The fused op runs forward on Q/K and its autograd backward launches the Triton
kernel with BACKWARD=True.  Inputs are (batch, heads, seq, head_dim) with
cos/sin (batch, seq, head_dim); the kernel covers the whole head dimension and
supports only float16/bfloat16 CUDA tensors.  Gradients are produced for both Q
and K; only Q's gradient is compared here.
"""
import statistics
import torch
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import kernelgenbench


COMBOS = [
    (dtype, batch, heads, kv_heads, seq, head_dim)
    for dtype in (torch.float16, torch.bfloat16)
    for (batch, heads, kv_heads, seq, head_dim) in (
        (2, 4, 4, 16, 64),
        (2, 3, 1, 11, 128),
        (1, 8, 2, 33, 64),
    )
]


def input_build(config):
    dtype, batch, heads, kv_heads, seq, head_dim = config
    gen = torch.Generator(device="cpu").manual_seed(141)
    grad = torch.randn((batch, heads, seq, head_dim), generator=gen, dtype=torch.float32).to(dtype)
    cos = torch.cos(torch.randn((batch, seq, head_dim), generator=gen, dtype=torch.float32)).to(dtype)
    sin = torch.sin(torch.randn((batch, seq, head_dim), generator=gen, dtype=torch.float32)).to(dtype)
    return {"grad_output": grad.to(kernelgenbench.device),
            "freqs_cos": cos.to(kernelgenbench.device),
            "freqs_sin": sin.to(kernelgenbench.device)}


def compute_ref(grad_output, freqs_cos, freqs_sin):
    return kernelgenbench.baseline.v1_ds_rotary_backward_triton(
        grad_output, freqs_cos, freqs_sin)


def compute_res(grad_output, freqs_cos, freqs_sin):
    return kernelgenbench.solution.v1_ds_rotary_backward_triton(
        grad_output, freqs_cos, freqs_sin)


def compare(ref_out, res_out, config):
    dtype, batch, heads, kv_heads, seq, head_dim = config
    assert res_out.shape == ref_out.shape == (batch, heads, seq, head_dim)
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = {torch.float16: (2e-3, 2e-3),
                  torch.bfloat16: (2e-2, 2e-2)}[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return statistics.median([_timed(compute_ref, inputs) for _ in range(30)])


def compute_res_performance(**inputs):
    return statistics.median([_timed(compute_res, inputs) for _ in range(30)])


def _timed(fn, inputs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn(**inputs)
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


@label("v1_ds_rotary_backward_triton")
@parametrize("config", COMBOS)
def test_accuracy_v1_ds_rotary_backward_triton(config):
    dtype, batch, heads, kv_heads, seq, head_dim = config
    inputs = input_build(config)
    assert inputs["grad_output"].shape == (batch, heads, seq, head_dim)
    assert inputs["freqs_cos"].shape == (batch, seq, head_dim)

    ref_out = compute_ref(**inputs)
    res_out = compute_res(**inputs)
    compare(ref_out, res_out, config)

    # Gradients are fresh tensors; the upstream kernel does not mutate cos/sin.
    assert not torch._C._is_alias_of(res_out, inputs["grad_output"])
    assert torch.equal(inputs["freqs_cos"].cpu(), inputs["freqs_cos"].detach().cpu())

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
