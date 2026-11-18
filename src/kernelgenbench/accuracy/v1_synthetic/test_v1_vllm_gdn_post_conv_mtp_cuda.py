"""vLLM fused GDN post-convolution MTP decode on the pinned CUDA kernel.

Bound to torch.ops._C.fused_gdn_decode_post_conv_mtp through
vllm._custom_ops.fused_gdn_decode_post_conv_mtp (the CUDA branch of
fused_mtp_decode_impl). Same contract as the Triton backend:
recurrence, per-token state writes, gated RMSNorm.

mixed_qkv packs [q|k|v] with K=V=128 per head; a/b carry the gate inputs and
state_indices[request, t] the destination cache slot per draft token.  The
recurrence starts from the slot of the last accepted token, writes the state
after every token, and finally applies the gated RMSNorm into `out`.  A slot
<= 0 means "no state": the whole request block is zeroed and no state is read.
Both `state` and `out` are written in place and the call returns None.
"""
import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

NAME = "v1_vllm_gdn_post_conv_mtp_cuda"
K = V = 128
# token counts per request, accepted counts, value heads per key head, state dtype,
# gate, dt dtype, norm dtype
CASES = [((3, 2), (1, 2), 1, torch.float32, "silu", torch.float32, torch.float32),
         ((1, 2), (1, 1), 2, torch.bfloat16, "sigmoid", torch.bfloat16, torch.bfloat16),
         ((2, 1), (0, 1), 3, torch.float32, "sigmoid", torch.float16, torch.float32),
         ((0, 1), (1, 1), 1, torch.bfloat16, "silu", torch.float32, torch.bfloat16),
         ((1,), (1,), 4, torch.float32, "silu", torch.float16, torch.bfloat16),
         ((2,), (1,), 8, torch.bfloat16, "sigmoid", torch.bfloat16, torch.float32)]
COMBOS = CASES


def input_build(config):
    lengths, accepted, hv, st, gate, dt, nt = config
    tokens = sum(lengths)
    nreq = len(lengths)
    rng = torch.Generator(device="cpu").manual_seed(tokens * 100 + hv)
    width = max(max(lengths), 2)
    # Slot 0 is the "no state" sentinel; request rows get distinct nonnegative slots.
    indices = torch.tensor([[1 + i * width + j for j in range(width)] for i in range(nreq)],
                           dtype=torch.int32)
    slots = 1 + nreq * width
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return dict(mixed_qkv=(torch.randn(tokens, (2 + hv) * K, generator=rng) * .2).bfloat16(),
                a=(torch.randn(tokens, hv, generator=rng) * .4).bfloat16(),
                b=(torch.randn(tokens, hv, generator=rng) * .4).bfloat16(),
                A_log=torch.randn(hv, generator=rng) * .2,
                dt_bias=(torch.randn(hv, generator=rng) * .2).to(dt),
                state_indices=indices, cu_seqlens=cu,
                num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32),
                state=(torch.randn(slots, hv, V, K, generator=rng) * .03).to(st),
                output_gate=torch.randn(tokens, hv, V, generator=rng).bfloat16(),
                norm_weight=(torch.rand(V, generator=rng) * .5 + .75).to(nt),
                out=torch.full((tokens, hv, V), 99, dtype=torch.bfloat16),
                scale=K ** -.5, norm_eps=1e-5, output_gate_activation=gate)


def _cuda(inputs):
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            moved = v.cuda()
            # a/b and mixed_qkv must stay contiguous in the layout the kernel asserts.
            out[k] = moved.contiguous() if k in ("mixed_qkv", "a", "b") else moved
        else:
            out[k] = v
    return out


def compute_ref(**kw):
    return kernelgenbench.baseline.v1_vllm_gdn_post_conv_mtp_cuda(**kw)


def compute_res(**kw):
    return kernelgenbench.solution.v1_vllm_gdn_post_conv_mtp_cuda(**kw)


def compare(ref_out, res_out, ref_in, res_in, ptrs):
    # Upstream writes `out` in place and returns that same buffer.
    assert ref_out is ref_in["out"], "reference must return the caller's out buffer"
    assert res_out is res_in["out"], "solution must return the caller's out buffer"
    assert res_in["out"].dtype == ref_in["out"].dtype == torch.bfloat16
    torch.testing.assert_close(res_in["out"], ref_in["out"], atol=.025, rtol=.03)
    torch.testing.assert_close(res_in["state"], ref_in["state"], atol=.0025, rtol=.03)
    # In-place contract: the caller's buffers keep their identity.
    assert (res_in["out"].data_ptr(), res_in["state"].data_ptr()) == ptrs


def _bench(fn, data):
    warm = _cuda(data)
    fn(**warm)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(20):
        fn(**_cuda(data))
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 20


def compute_ref_performance(inputs):
    return _bench(compute_ref, inputs)


def compute_res_performance(inputs):
    return _bench(compute_res, inputs)


@label(NAME)
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_gdn_post_conv_mtp_cuda(config):
    inputs = input_build(config)
    ref_in = _cuda(inputs)
    res_in = _cuda(inputs)
    ptrs = (res_in["out"].data_ptr(), res_in["state"].data_ptr())
    untouched = ("mixed_qkv", "a", "b", "A_log", "dt_bias", "state_indices",
                 "cu_seqlens", "num_accepted_tokens", "output_gate", "norm_weight")
    saved = {k: res_in[k].clone() for k in untouched}
    ref_out = compute_ref(**ref_in)
    res_out = compute_res(**res_in)
    compare(ref_out, res_out, ref_in, res_in, ptrs)
    for name in untouched:
        torch.testing.assert_close(res_in[name], saved[name], rtol=0, atol=0)
    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
