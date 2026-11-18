# Synthetic workload; no production capture or production frequency claim.
"""OP-66a70cb956cc (Megatron branch): _apply_rotary_pos_emb_bshd.

t is [seq, batch, heads, head_dim] and freqs is [seq, batch, 1, rot_dim] real
angular frequencies; the last rot_dim features rotate (split-half or
interleaved pairing, optional MLA reordering and mscale) and the rest pass
through.  Only the official NVIDIA entry is bound; the fork-only ``inverse``
flag is not part of this item.
"""
import statistics
import torch
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import kernelgenbench


COMBOS = [
    (dtype, seq, batch, heads, head_dim, rot_dim, interleaved, mla, mscale)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for (seq, batch, heads, head_dim, rot_dim, interleaved, mla, mscale) in (
        (16, 2, 4, 64, 64, False, False, 1.0),
        (16, 2, 4, 64, 32, False, False, 1.0),
        (16, 2, 4, 64, 64, True, False, 1.0),
        (9, 1, 2, 128, 64, False, True, 1.0),
        (9, 1, 2, 64, 64, False, False, 1.2),
    )
]


def input_build(config):
    dtype, seq, batch, heads, head_dim, rot_dim, interleaved, mla, mscale = config
    gen = torch.Generator(device="cpu").manual_seed(141)
    t = torch.randn((seq, batch, heads, head_dim), generator=gen, dtype=torch.float32).to(dtype)
    freqs = torch.randn((seq, batch, 1, rot_dim), generator=gen, dtype=torch.float32)
    return {"t": t.to(kernelgenbench.device), "freqs": freqs.to(kernelgenbench.device),
            "rotary_interleaved": interleaved, "multi_latent_attention": mla,
            "mscale": mscale}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_rope_bshd(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_rope_bshd(**inputs)


def compare(ref_out, res_out, config):
    dtype, seq, batch, heads, head_dim, rot_dim, interleaved, mla, mscale = config
    assert res_out.shape == ref_out.shape == (seq, batch, heads, head_dim)
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = {torch.float32: (2e-6, 2e-6),
                  torch.float16: (2e-3, 2e-3),
                  torch.bfloat16: (2e-2, 2e-2)}[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def _time(fn, inputs, reps=200):
    for _ in range(20):
        fn(**inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn(**inputs)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / reps


@label("v1_rope_bshd")
@parametrize("config", COMBOS)
def test_accuracy_v1_rope_bshd(config):
    dtype, seq, batch, heads, head_dim, rot_dim, interleaved, mla, mscale = config
    inputs = input_build(config)
    assert inputs["t"].shape == (seq, batch, heads, head_dim)
    assert inputs["freqs"].shape == (seq, batch, 1, rot_dim)

    ref_out = compute_ref(**inputs)
    res_out = compute_res(**inputs)
    compare(ref_out, res_out, config)

    # Functional: neither input is mutated and the result is fresh.
    assert not torch._C._is_alias_of(res_out, inputs["t"])
    assert torch.equal(inputs["t"].cpu(), inputs["t"].detach().cpu())
    assert torch.equal(inputs["freqs"].cpu(), inputs["freqs"].detach().cpu())
    # The non-rotary tail is copied verbatim.
    if rot_dim < head_dim:
        torch.testing.assert_close(res_out[..., rot_dim:], inputs["t"][..., rot_dim:],
                                   rtol=0, atol=0)
    ref_time = _time(compute_ref, inputs)
    res_time = _time(compute_res, inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
