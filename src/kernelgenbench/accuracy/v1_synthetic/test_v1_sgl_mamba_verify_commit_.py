"""SGLang post-verify Mamba state commit: accepted-step scatter into persistent caches.

temporal is [layers, slots, H, P, N]; intermediate_ssm is the per-request verify
scratch [layers, requests, draft, H, P, N]; conv_states is one [layers, slots,
dim, K-1] pool per conv type and intermediate_conv_window the matching
[layers, requests, draft, dim, K-1] tape (optionally an overlapping as_strided
view of a shared [dim, draft+K-2] buffer).  For each request the state of the
last accepted step is written to state_indices[request] (and, when tracking is
on, the interval-crossing step to mamba_track_indices[request]); slot <= 0 is a
sentinel and is skipped.  The call only mutates the caches and returns None.
"""
import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

NAME = "v1_sgl_mamba_verify_commit_"
DEV = "cuda"
# requests, draft tokens, layers, pool size, tracking, stable scratch rows,
# prepared indices, number of conv types, dtype, freed destination slot
CASES = [(3, 4, 2, 8, True, False, False, 2, torch.bfloat16, False),
         (2, 4, 1, 6, False, True, False, 1, torch.float32, False),
         (3, 5, 2, 9, True, True, True, 2, torch.bfloat16, False),
         (3, 4, 2, 8, False, False, False, 1, torch.float32, True),
         (4, 4, 2, 10, True, False, False, 0, torch.float32, True),
         (0, 4, 1, 5, False, False, False, 1, torch.float32, False)]
COMBOS = CASES


def input_build(config):
    bs, draft, layers, pool, track, stable, prepared, nconv, dtype, freed = config
    scratch = max(bs, 1)
    ssm_shape = (layers, pool, 2, 3)
    src_shape = (layers, scratch, draft, 2, 3)
    temporal = torch.full(ssm_shape, -77, dtype=dtype, device=DEV)
    intermediate = (torch.arange(torch.tensor(src_shape).prod().item(), device=DEV,
                                 dtype=torch.float32) % 251).to(dtype).reshape(src_shape)
    conv, windows = [], []
    for j in range(nconv):
        dim, win = 3 + j, 2 + j
        dst = torch.full((layers, pool, dim, win), -87 - j, dtype=dtype, device=DEV)
        raw = (torch.arange(layers * scratch * dim * (draft + win - 1), device=DEV,
                            dtype=torch.float32) + 50 * j).to(dtype).reshape(
                                layers, scratch, dim, draft + win - 1)
        # Overlapping tape: step t's window is shared[:, t:t+win].
        window = raw.as_strided((layers, scratch, draft, dim, win),
                                (raw.stride(0), raw.stride(1), 1, raw.stride(2), 1))
        conv.append(dst)
        windows.append(window)
    lens = torch.tensor([1, 3, 2, 4][:bs], device=DEV, dtype=torch.int32)
    rows = torch.arange(bs, device=DEV, dtype=torch.int64)
    relative = (torch.stack([torch.roll(torch.arange(draft, device=DEV), shifts=i % draft)
                             for i in range(bs)])
                if bs else torch.empty((0, draft), dtype=torch.int64, device=DEV))
    accept_index = relative + rows[:, None] * draft
    seq_lens = torch.tensor([3, 4, 6, 7][:bs], device=DEV, dtype=torch.int32)
    states = torch.arange(bs, device=DEV, dtype=torch.int64)
    if freed and bs:
        states[-1] = -1
    tracking = (pool - 1 - torch.arange(bs, device=DEV, dtype=torch.int64)) if track else None
    source = torch.tensor(([2, 0, 1, 3][:bs] if bs > 2 else [1, 0][:bs]),
                          device=DEV, dtype=torch.int64) if stable else None
    prep = None
    if prepared:
        last = accept_index[rows, (lens - 1).long()] - rows * draft
        post = seq_lens + lens
        which = (post // 4 * 4 - seq_lens - 1).clamp(min=0).long()
        track_steps = (torch.where(seq_lens // 4 != post // 4,
                                   accept_index[rows, which] - rows * draft, -1)
                       if track else None)
        prep = (last, track_steps)
    return dict(temporal=temporal, intermediate_ssm=intermediate,
                conv_states=tuple(conv), intermediate_conv_window=tuple(windows),
                accept_lens=lens, accept_index=accept_index, seq_lens=seq_lens,
                draft_token_num=draft, state_indices=states,
                mamba_track_indices=tracking, track_interval=4 if track else 0,
                src_indices_raw=source, prepared_step_indices=prep)


def compute_ref(**kw):
    return kernelgenbench.baseline.v1_sgl_mamba_verify_commit_(**kw)


def compute_res(**kw):
    return kernelgenbench.solution.v1_sgl_mamba_verify_commit_(**kw)


def compare(ref_out, res_out, ref_in, res_in):
    # Both entries only mutate the caches and return None.
    assert ref_out is None and res_out is None
    assert torch.equal(res_in["temporal"], ref_in["temporal"])
    assert len(res_in["conv_states"]) == len(ref_in["conv_states"])
    for a, b in zip(ref_in["conv_states"], res_in["conv_states"]):
        assert torch.equal(a, b)
    # Verify scratch and tapes are read-only inputs.
    assert torch.equal(res_in["intermediate_ssm"], ref_in["intermediate_ssm"])
    for a, b in zip(ref_in["intermediate_conv_window"], res_in["intermediate_conv_window"]):
        assert torch.equal(a, b)


def _bench(fn, inputs):
    fn(**inputs)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(20):
        fn(**inputs)
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
def test_accuracy_v1_sgl_mamba_verify_commit_(config):
    inputs = input_build(config)
    saved = {k: ([x.clone() for x in v] if isinstance(v, tuple) else
                 v.clone() if isinstance(v, torch.Tensor) else v)
             for k, v in inputs.items()}
    ref_in = dict(inputs)
    res_in = dict(inputs)
    # Each side gets its own copy of the caches that are written in place.
    for side in (ref_in, res_in):
        side["temporal"] = inputs["temporal"].clone()
        side["conv_states"] = tuple(x.clone() for x in inputs["conv_states"])
    ptrs = ([x.data_ptr() for x in res_in["conv_states"]], res_in["temporal"].data_ptr())
    ref_out = compute_ref(**ref_in)
    res_out = compute_res(**res_in)
    compare(ref_out, res_out, ref_in, res_in)
    # Cache buffers keep their identity; nothing else is written.
    assert [x.data_ptr() for x in res_in["conv_states"]] == ptrs[0]
    assert res_in["temporal"].data_ptr() == ptrs[1]
    for name in ("intermediate_ssm", "accept_lens", "accept_index", "seq_lens",
                 "state_indices", "mamba_track_indices", "src_indices_raw"):
        if isinstance(saved[name], torch.Tensor):
            assert torch.equal(saved[name], inputs[name]), name
    for name in ("conv_states", "intermediate_conv_window"):
        for a, b in zip(saved[name], inputs[name]):
            assert torch.equal(a, b), name
    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
