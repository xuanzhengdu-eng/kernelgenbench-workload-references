# Synthetic workload; no production capture or production frequency claim.
"""OP-8b62b7b19686 (fpdt branch): DeepSpeed apply_rotary_pos_emb_backward.

grad_output is (batch, heads, seq, head_dim); freqs_cos/freqs_sin broadcast as
(batch, 1, seq, rot_dim).  The first rot_dim features rotate with split-half
pairing and the tail passes through, preserving dtype and returning a new
tensor.  Odd or too-wide rotary widths raise ValueError.
"""
import statistics
import torch
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import kernelgenbench


COMBOS = [
    (dtype, batch, heads, seq, head_dim, rot_dim)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for (batch, heads, seq, head_dim, rot_dim) in (
        (2, 4, 16, 64, 64),
        (2, 3, 11, 64, 32),
        (1, 2, 7, 128, 64),
        (3, 1, 5, 32, 8),
    )
]


def input_build(config):
    dtype, batch, heads, seq, head_dim, rot_dim = config
    gen = torch.Generator(device="cpu").manual_seed(140)
    grad = torch.randn((batch, heads, seq, head_dim), generator=gen, dtype=torch.float32).to(dtype)
    cos = torch.cos(torch.randn((batch, 1, seq, rot_dim), generator=gen, dtype=torch.float32)).to(dtype)
    sin = torch.sin(torch.randn((batch, 1, seq, rot_dim), generator=gen, dtype=torch.float32)).to(dtype)
    return {"grad_output": grad.to(kernelgenbench.device),
            "freqs_cos": cos.to(kernelgenbench.device),
            "freqs_sin": sin.to(kernelgenbench.device)}


def compute_ref(grad_output, freqs_cos, freqs_sin):
    return kernelgenbench.baseline.v1_ds_rotary_backward_fpdt(
        grad_output, freqs_cos, freqs_sin)


def compute_res(grad_output, freqs_cos, freqs_sin):
    return kernelgenbench.solution.v1_ds_rotary_backward_fpdt(
        grad_output, freqs_cos, freqs_sin)


def compare(ref_out, res_out, config):
    dtype, batch, heads, seq, head_dim, rot_dim = config
    assert res_out.shape == ref_out.shape == (batch, heads, seq, head_dim)
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = {torch.float32: (1e-6, 1e-6),
                  torch.float16: (2e-3, 2e-3),
                  torch.bfloat16: (2e-2, 2e-2)}[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(res_out[..., rot_dim:], ref_out[..., rot_dim:],
                               rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return statistics.median([
        _timed(compute_ref, inputs) for _ in range(30)])


def compute_res_performance(**inputs):
    return statistics.median([
        _timed(compute_res, inputs) for _ in range(30)])


def _timed(fn, inputs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn(**inputs)
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


@label("v1_ds_rotary_backward_fpdt")
@parametrize("config", COMBOS)
def test_accuracy_v1_ds_rotary_backward_fpdt(config):
    dtype, batch, heads, seq, head_dim, rot_dim = config
    inputs = input_build(config)
    assert inputs["grad_output"].shape == (batch, heads, seq, head_dim)
    assert inputs["grad_output"].dtype == dtype

    ref_out = compute_ref(**inputs)
    res_out = compute_res(**inputs)
    compare(ref_out, res_out, config)

    # Pure function over its arguments: nothing is mutated and the result is new.
    for name, tensor in inputs.items():
        assert not torch._C._is_alias_of(res_out, tensor), f"output aliases {name}"
    assert torch.equal(inputs["freqs_cos"].cpu(), inputs["freqs_cos"].detach().cpu())

    # Documented invalid widths: odd rotary dimension and wider than the input.
    try:
        compute_ref(inputs["grad_output"], inputs["freqs_cos"][..., : rot_dim - 1],
                    inputs["freqs_sin"][..., : rot_dim - 1])
    except ValueError as exc:
        assert "rotary dimension" in str(exc)
    else:
        raise AssertionError("odd rotary width must raise ValueError")

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
