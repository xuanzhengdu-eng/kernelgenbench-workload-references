# Synthetic workload; no production capture or production frequency claim.
"""OP-0c1a0899dfff: Megatron MultimodalRotaryEmbedding.forward.

position_ids is (3, batch, seq) int64 T/H/W; inv_freq is (rotary_dim//2,);
mrope_section sums to rotary_dim//2.  The output is (seq, batch, 1,
rotary_dim) float32, sequence-major.  Only the official NVIDIA entry is
bound: the fork-only interleaved_mrope / per-frequency interleaving branch is
removed from this item.
"""
import statistics
import torch
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import kernelgenbench


COMBOS = [
    (batch, seq, head_size, rotary_percent, section, rotary_interleaved, factor, pos_dtype)
    for batch in (1, 3)
    for seq in (5, 32)
    for (head_size, rotary_percent, section, rotary_interleaved, factor, pos_dtype) in (
        (64, 1.0, [8, 8, 16], False, None, torch.int64),
        (64, 0.5, [8, 4, 4], False, None, torch.int64),
        (128, 0.25, [8, 4, 4], False, 2.0, torch.int64),
        (64, 1.0, [8, 8, 16], False, 2.0, torch.float32),
        (64, 0.5, [8, 4, 4], False, 3.0, torch.float32),
    )
    # rotary_interleaved=True is excluded: at the pinned NVIDIA commit the
    # upstream branch calls .view() on a transposed tensor and raises
    # "view size is not compatible ... Use .reshape()".  That is an upstream
    # defect, not a KernelGenBench omission; evidence in
    # logs/row139/interleaved_upstream_error.log.
]


def input_build(config):
    batch, seq, head_size, rotary_percent, section, rotary_interleaved, factor, pos_dtype = config
    gen = torch.Generator(device="cpu").manual_seed(139)
    rotary_dim = int(head_size * rotary_percent)
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    assert len(inv_freq) == sum(section)
    seq_ids = torch.arange(seq)
    position_ids = torch.stack(
        [seq_ids % 7 + 1, seq_ids % 5, seq_ids % 3 + 2], dim=0
    ).to(pos_dtype)[:, None, :].expand(3, batch, seq).contiguous()
    return {"position_ids": position_ids.to(kernelgenbench.device),
            "inv_freq": inv_freq.to(kernelgenbench.device),
            "mrope_section": list(section),
            "rotary_interleaved": rotary_interleaved,
            "seq_len_interpolation_factor": factor,
            "cp_size": 1, "cp_rank": 0}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_mrope_freqs(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_mrope_freqs(**inputs)


def compare(ref_out, res_out, config):
    batch, seq, head_size, rotary_percent, section, rotary_interleaved, factor, pos_dtype = config
    rotary_dim = int(head_size * rotary_percent)
    # inv_freq has rotary_dim//2 entries; upstream emits 2*dim = rotary_dim columns.
    assert res_out.shape == ref_out.shape == (seq, batch, 1, rotary_dim)
    assert res_out.dtype == ref_out.dtype == torch.float32
    torch.testing.assert_close(res_out, ref_out, rtol=2e-6, atol=2e-6)


def _time(fn, inputs, warmup=20, reps=100):
    for _ in range(warmup):
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


@label("v1_megatron_mrope_freqs")
@parametrize("config", COMBOS)
def test_accuracy_v1_megatron_mrope_freqs(config):
    batch, seq, head_size, rotary_percent, section, rotary_interleaved, factor, pos_dtype = config
    inputs = input_build(config)
    assert inputs["position_ids"].shape == (3, batch, seq)
    assert sum(section) == len(inputs["inv_freq"])

    original = inputs["position_ids"].float().cpu()

    def clones():
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()}

    ref_inputs = clones()
    res_inputs = clones()
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    # Determinism and input handling.  With a float32 position_ids and a
    # scaling factor the upstream conversion is a no-op and the in-place
    # ``seq *= 1/factor`` updates the caller's tensor; an integer input is
    # copied first and stays unchanged.  The pristine inputs are only ever
    # handed out as clones.
    assert torch.equal(compute_res(**clones()), res_out)
    assert torch.equal(inputs["position_ids"].float().cpu(), original)
    scales_in_place = factor is not None and pos_dtype == torch.float32
    for side in (ref_inputs, res_inputs):
        moved = not torch.equal(side["position_ids"].float().cpu(), original)
        assert moved == scales_in_place, (pos_dtype, factor, moved)
        if scales_in_place:
            torch.testing.assert_close(
                side["position_ids"].float().cpu(),
                original / factor, rtol=1e-6, atol=1e-6)
    # Upstream returns raw angular frequencies (cos/sin are applied later by
    # rope_utils).  ``mrope_section * 2`` repeats the three widths, so the split
    # order is T,H,W,T,H,W: axis 1 starts at column section[0] and axis 2 at
    # section[0] + section[1].
    scale = 1.0 if factor is None else 1.0 / factor
    first = inputs["inv_freq"][0].float()
    expected_t = inputs["position_ids"][0].float() * scale * first
    torch.testing.assert_close(res_out[..., 0].squeeze(-1).transpose(0, 1),
                               expected_t, rtol=2e-6, atol=2e-6)
    n_freq = len(inputs["inv_freq"])
    for axis, column in ((1, section[0]), (2, section[0] + section[1])):
        if section[axis] == 0:
            continue
        rate = inputs["inv_freq"][column % n_freq].float()
        expected_axis = inputs["position_ids"][axis].float() * scale * rate
        torch.testing.assert_close(res_out[..., column].squeeze(-1).transpose(0, 1),
                                   expected_axis, rtol=2e-6, atol=2e-6)

    ref_time = _time(compute_ref, inputs)
    res_time = _time(compute_res, inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
