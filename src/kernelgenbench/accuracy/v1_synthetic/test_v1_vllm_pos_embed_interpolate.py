# Synthetic workload; no production capture or production frequency claim.
"""OP-aa68153dafa5: vLLM positional-embedding bilinear interpolation.

embed_weight is (num_grid_per_side**2, hidden).  Output is
(t*h*w, hidden): each grid is bilinearly sampled, rearranged into
spatial-merge order (m_size x m_size tiles), then repeated t times.
The solution binds the fused Triton kernel that vLLM dispatches to on CUDA.
"""
import torch
import triton.testing
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import kernelgenbench


COMBOS = [
    (dtype, t, h, w, side, m_size, hidden)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for (t, h, w, side, m_size, hidden) in (
        (1, 4, 6, 12, 2, 64),
        (2, 6, 4, 12, 2, 64),
        (1, 2, 2, 12, 2, 64),
        (3, 8, 8, 16, 4, 64),
        (1, 4, 4, 9, 2, 64),
    )
]


def input_build(config):
    dtype, t, h, w, side, m_size, hidden = config
    gen = torch.Generator(device="cpu").manual_seed(135)
    weight = torch.randn((side * side, hidden), generator=gen, dtype=torch.float32).to(dtype)
    return {"embed_weight": weight.to(kernelgenbench.device), "t": t, "h": h, "w": w,
            "num_grid_per_side": side, "m_size": m_size, "dtype": dtype}


def compute_ref(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
    return kernelgenbench.baseline.v1_vllm_pos_embed_interpolate(
        embed_weight, t, h, w, num_grid_per_side, m_size, dtype)


def compute_res(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
    return kernelgenbench.solution.v1_vllm_pos_embed_interpolate(
        embed_weight, t, h, w, num_grid_per_side, m_size, dtype)


def compare(ref_out, res_out, config):
    dtype, t, h, w, side, m_size, hidden = config
    assert res_out.shape == ref_out.shape == (t * h * w, hidden)
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = {torch.float32: (1e-5, 2e-5),
                  torch.float16: (2e-3, 2e-3),
                  torch.bfloat16: (3e-2, 3e-2)}[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_vllm_pos_embed_interpolate")
@parametrize("config", COMBOS)
def test_accuracy_v1_vllm_pos_embed_interpolate(config):
    dtype, t, h, w, side, m_size, hidden = config
    inputs = input_build(config)
    assert inputs["embed_weight"].shape == (side * side, hidden)
    assert inputs["embed_weight"].dtype == dtype
    assert h % m_size == 0 and w % m_size == 0

    ref_out = compute_ref(**inputs)
    res_out = compute_res(**inputs)
    compare(ref_out, res_out, config)

    # The lookup is pure: the weight table is not modified, and the result is
    # a fresh contiguous tensor.
    assert inputs["embed_weight"].dtype == dtype
    assert res_out.is_contiguous()
    assert not torch._C._is_alias_of(res_out, inputs["embed_weight"])

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
