"""Accuracy and performance for v1_megatron_mla_full.

Operator: Megatron MLA forward with q_lora_rank=None (q/kv projections, kv
RMSNorm, even/odd RoPE, masked softmax attention, output projection) and a
separate output bias that is returned but not added.
Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/multi_latent_attention.py#L216-L374
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/multi_latent_attention.py#L553-L730
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/rope_utils.py#L92-L126

Contract: FP32 x [S,B,I]; q weight [H*(C+P),I]; KV down weight [R+P,I]; RMS
weight [R]; KV up weight [H*(C+V),R]; out weight [O,H*V]; separate out bias
[O]; frequency table [S,1,1,P] (angles, cos/sin applied inside); bool mask
[B,1,S,S] with True masked to -10000 before softmax.  P must be positive and
even.  Scores use (C+P)**-0.5.  The return is ``(output [S,B,O], bias [O])``:
the bias is the separate projection bias, it is *not* added to the output.  The
former absorbed/MockCoreAttention branch is removed (no official operator).

Item: v1_megatron_mla_full (723-row 271).
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.upstream_timing import measure
from sandbox.verifier.test_parametrize import label, parametrize


# (S, B, I, H, C, P, V, R, O, mask_mode)
# mask_mode "causal" masks the strict upper triangle; "mixed" also masks one
# extra full column so a masked key is reachable from every row.
COMBOS = [
    (3, 1, 8, 2, 2, 2, 4, 4, 8, "causal"),
    (4, 2, 8, 2, 2, 2, 4, 6, 8, "causal"),
    (3, 2, 6, 1, 2, 2, 4, 4, 6, "causal"),
    (4, 1, 8, 2, 2, 4, 6, 4, 8, "causal"),
    (5, 2, 8, 4, 4, 2, 4, 8, 8, "causal"),
    (6, 1, 8, 2, 2, 2, 4, 4, 8, "mixed"),
    (2, 3, 4, 2, 2, 2, 2, 4, 4, "causal"),
]

TOLERANCE = (2e-5, 1e-6)


def _values(shape, multiplier):
    """Deterministic bounded FP32 values on CPU."""
    count = 1
    for size in shape:
        count *= size
    flat = (((torch.arange(count, dtype=torch.float32) * multiplier) % 47) - 23) / 50
    return flat.reshape(shape)


def input_build(config):
    """Build the projection weights, frequency table and bool mask.

    freq is [S,1,1,P] with angle = position / 7 * (pair index), duplicated for
    the two halves so the RoPE table has the full head width.  The mask is
    [B,1,S,S] bool; True marks a masked position.
    """
    S, B, I, H, C, P, V, R, O, mode = config
    angles = (torch.arange(S, dtype=torch.float32)[:, None]
              * torch.arange(1, P // 2 + 1, dtype=torch.float32)[None, :]) / 7
    freq = torch.cat((angles, angles), -1).reshape(S, 1, 1, P)
    mask = torch.ones((B, 1, S, S), dtype=torch.bool).triu(1)
    if mode == "mixed":
        mask[:, :, :, S - 1] = True
        mask[:, :, S - 1, S - 1] = False
    return dict(
        x=_values((S, B, I), 3), wq=_values((H * (C + P), I), 5),
        wkv_down=_values((R + P, I), 7), rms_weight=_values((R,), 11) + 1.0,
        wkv_up=_values((H * (C + V), R), 13), wout=_values((O, H * V), 17),
        out_bias=_values((O,), 19), freq=freq, mask=mask, heads=H,
        content_dim=C, rope_dim=P, value_dim=V, eps=1e-5)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_mla_full(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_mla_full(**inputs)


def compare(ref_out, res_out, config):
    """Output [S,B,O] fp32 plus the separate, unadded bias [O]."""
    S, B, O = config[0], config[1], config[8]
    assert isinstance(res_out, (tuple, list)) and len(res_out) == 2, \
        f"expected (output, bias), got {type(res_out).__name__}"
    ref_y, ref_b = ref_out
    res_y, res_b = res_out
    assert res_y.shape == (S, B, O), f"shape {tuple(res_y.shape)} != {(S, B, O)}"
    assert res_y.dtype == ref_y.dtype == torch.float32
    assert res_y.device == ref_y.device and res_y.is_contiguous()
    rtol, atol = TOLERANCE
    torch.testing.assert_close(res_y, ref_y, rtol=rtol, atol=atol)
    assert isinstance(res_b, torch.Tensor) and res_b.shape == (O,)
    torch.testing.assert_close(res_b, ref_b, rtol=0, atol=0)
    assert not torch._C._is_alias_of(res_y, res_b), "bias aliases the output"
    assert torch.isfinite(res_y).all()


def compute_ref_performance(**inputs):
    return measure(compute_ref, kwargs=inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, kwargs=inputs)


@label("v1_megatron_mla_full")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_mla_full(config):
    inputs = input_build(config)
    device = kernelgenbench.device
    S, B, I, H, C, P, V, R, O, mode = config
    assert inputs["x"].shape == (S, B, I)
    assert inputs["wq"].shape == (H * (C + P), I)
    assert inputs["wkv_down"].shape == (R + P, I)

    def move(values):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in values.items()}

    ref_inputs, res_inputs = move(inputs), move(inputs)
    snapshots = {k: v.clone() for k, v in res_inputs.items() if isinstance(v, torch.Tensor)}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)
    for key, value in snapshots.items():
        torch.testing.assert_close(res_inputs[key], value, rtol=0, atol=0)
    assert not torch._C._is_alias_of(res_out[0], res_inputs["x"])

    # The separate bias must not enter the output: doubling it changes only the
    # returned bias, not the projected result.
    doubled = dict(res_inputs)
    doubled["out_bias"] = res_inputs["out_bias"] * 2
    res_y2, res_b2 = compute_res(**doubled)
    torch.testing.assert_close(res_y2, res_out[0], rtol=0, atol=0)
    torch.testing.assert_close(res_b2, res_inputs["out_bias"] * 2, rtol=0, atol=0)

    device_inputs = move(inputs)
    ref_time = compute_ref_performance(**device_inputs)
    res_time = compute_res_performance(**device_inputs)
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult

    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
