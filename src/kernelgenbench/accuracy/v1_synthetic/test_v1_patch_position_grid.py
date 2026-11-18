# Synthetic workload; no production capture or production frequency claim.
"""OP-2d9ba80fb7b5: SGLang Pixtral position_meshgrid.

Each COMBOS entry is a tuple of patch-grid shapes.  Inputs are (C, H, W)
tensors; the reference ignores their values and reads only H and W.  Output is
one int64 CPU (row, column) pair per patch, concatenated in list order.
"""
import time
import torch
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import kernelgenbench


COMBOS = [
    (((3, 4),),),
    (((2, 5), (1, 1)),),
    (((1, 7), (4, 2), (3, 3)),),
    (((0, 5), (2, 0)),),
    (((5, 1),),),
]


def input_build(config):
    grids = config[0]
    gen = torch.Generator(device="cpu").manual_seed(132)
    return {
        "patch_embeds_list": [
            torch.randn((3, h, w), generator=gen, dtype=torch.float32) for h, w in grids
        ]
    }


def compute_ref(patch_embeds_list):
    return kernelgenbench.baseline.v1_patch_position_grid(patch_embeds_list)


def compute_res(patch_embeds_list):
    return kernelgenbench.solution.v1_patch_position_grid(patch_embeds_list)


def compare(ref_out, res_out):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape, (res_out.shape, ref_out.shape)
    assert res_out.dtype == ref_out.dtype == torch.int64
    assert res_out.device.type == ref_out.device.type == "cpu"
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def _time_cpu(fn, kwargs, warmup=10, rep=50):
    for _ in range(warmup):
        fn(**kwargs)
    samples = []
    for _ in range(rep):
        start = time.perf_counter()
        fn(**kwargs)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


@label("v1_patch_position_grid")
@parametrize("config", COMBOS)
def test_accuracy_v1_patch_position_grid(config):
    (grids,) = config
    inputs = input_build(config)
    assert [tuple(x.shape[-2:]) for x in inputs["patch_embeds_list"]] == list(grids)

    ref_out = compute_ref(**{k: list(v) for k, v in inputs.items()})
    res_out = compute_res(**{k: list(v) for k, v in inputs.items()})
    compare(ref_out, res_out)

    total = sum(h * w for h, w in grids)
    assert res_out.shape == (total, 2)
    if total:
        expected_h = torch.cat([torch.arange(h, dtype=torch.int64).repeat_interleave(w)
                                for h, w in grids])
        expected_w = torch.cat([torch.arange(w, dtype=torch.int64).repeat(h)
                                for h, w in grids])
        torch.testing.assert_close(res_out[:, 0], expected_h, rtol=0, atol=0)
        torch.testing.assert_close(res_out[:, 1], expected_w, rtol=0, atol=0)

    ref_time = _time_cpu(compute_ref, inputs)
    res_time = _time_cpu(compute_res, inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
