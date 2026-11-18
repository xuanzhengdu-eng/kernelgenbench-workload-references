# NVIDIA Megatron-LM pinned fused collective: multimem reduce-scatter +
# residual add + RMSNorm + all-gather into the symmetric-memory buffer.
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/communication/torch_symm_triton/fused_collectives.py#L210-L280
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/inference_layers.py#L420-L435
"""Fused multimem RS + residual add + RMSNorm + all-gather (Megatron, bf16).

Reference: ``kernelgenbench.baseline.v1_megatron_rs_add_rms_ag_`` (plain-torch
model of the same contract).
Solution: the pinned official
``megatron.core.inference.communication.torch_symm_triton.fused_multimem_rs_add_norm_ag``
bound to a real ``torch.distributed._symmetric_memory`` buffer handle, exactly as
``inference_layers.py#L420-L435`` calls it.

Contract under test:
  * both sides return ``residual_output_tensor`` and write the group-reduced
    residual sum into it (bf16);
  * the symmetric-memory buffer is overwritten in place with the normalized,
    all-gathered slices in rank order;
  * RMSNorm weights and eps are applied per row; nothing else is aliased.

Hardware precondition: the official entry launches a multimem Triton kernel, so
``symm_mem_hdl.multicast_ptr`` and ``signal_pad_ptrs_dev`` must be real device
addresses (NVLS multicast over distinct GPUs).  Each rank owns a separate physical GPU.
"""
from types import SimpleNamespace

import datetime
import json
import os
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# (world, tokens per rank, hidden size, residual input/output alias, epsilon)
COMBOS = [
    (2, 4, 128, False, 1e-5),
    (2, 8, 64, True, 1e-6),
    (2, 2, 128, False, 1e-5),
    (2, 4, 64, True, 1e-6),
]


def input_build(config, rank=0):
    world, local, hidden, alias, eps = config
    generator = torch.Generator().manual_seed(4051 + rank)
    partial = (torch.randn((world * local, hidden), generator=generator)
               .to(torch.bfloat16))
    residual = (torch.randn((local, hidden), generator=generator)
                .to(torch.bfloat16))
    weights = (torch.randn((hidden,), generator=torch.Generator().manual_seed(4110))
               .to(torch.bfloat16))
    output = residual if alias else torch.full_like(residual, -9)
    return {
        "residual_output_tensor": output,
        "input_tensor": partial,
        "residual_input_tensor": residual,
        "rms_norm_weights": weights,
        "eps": eps,
    }


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_megatron_rs_add_rms_ag_(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.triton.v1_megatron_rs_add_rms_ag_(**kwargs)


def compare(ref_out, res_out, ref_inputs, res_inputs, config):
    world, local, hidden, alias, eps = config
    assert res_out is res_inputs["residual_output_tensor"]
    assert ref_out is ref_inputs["residual_output_tensor"]
    assert res_out.dtype == torch.bfloat16 and res_out.shape == (local, hidden)
    torch.testing.assert_close(res_out, ref_out, rtol=2e-2, atol=2e-2)
    # Both sides publish the gathered normalized slices into the buffer.
    torch.testing.assert_close(
        res_inputs["symm_mem_buffer"], ref_inputs["input_tensor"],
        rtol=5e-2, atol=5e-2,
    )
    assert torch.equal(res_inputs["rms_norm_weights"], ref_inputs["rms_norm_weights"])


def _open_symmetric_buffer(world, local, hidden, rank):
    """Allocate the pinned symmetric-memory buffer and rendezvous the group."""
    import torch.distributed._symmetric_memory as symm_mem

    nbytes = world * local * hidden * 2
    raw = symm_mem.empty(nbytes, dtype=torch.uint8, device="cuda")
    handle = symm_mem.rendezvous(raw, dist.group.WORLD)
    buffer = raw.view(torch.bfloat16).view(world * local, hidden)
    return handle, buffer


def worker(rank, config, rendezvous, outdir, candidate_bytes):
    # Spawned ranks need the same registered Solution as the parent verifier.
    import cloudpickle
    kernelgenbench.triton = SimpleNamespace(v1_megatron_rs_add_rms_ag_=cloudpickle.loads(candidate_bytes))
    world, local, hidden, alias, eps = config
    if world > torch.cuda.device_count():
        raise RuntimeError("symmetric-memory collectives require one visible GPU per rank")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world,
        device_id=torch.device("cuda", rank),
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        handle, buffer = _open_symmetric_buffer(world, local, hidden, rank)
        assert handle.world_size == world
        res_inputs = _to_device(input_build(config, rank), "cuda")
        ref_inputs = _to_device(input_build(config, rank), "cuda")
        if alias:
            for side in (ref_inputs, res_inputs):
                side["residual_output_tensor"] = side["residual_input_tensor"]
        assert handle.multicast_ptr, "NVLink multicast allocation returned no pointer"
        # Both sides must start from identical symmetric-buffer contents.
        buffer.copy_(res_inputs["input_tensor"])
        ref_inputs["input_tensor"].copy_(buffer)

        ref_kwargs = dict(ref_inputs, group=dist.group.WORLD)
        ref_out = compute_ref(**ref_kwargs)
        res_out = compute_res(
            **dict(res_inputs, symm_mem_handle=handle, symm_mem_buffer=buffer)
        )
        torch.cuda.synchronize()
        dist.barrier()
        compare(ref_out, res_out, ref_inputs, dict(res_inputs, symm_mem_buffer=buffer), config)

        def bench(fn, kwargs):
            for _ in range(2):
                fn(**kwargs)
            torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            for _ in range(5):
                fn(**kwargs)
            torch.cuda.synchronize()
            dist.barrier()
            return (time.perf_counter() - start) * 1000.0 / 5.0

        ref_time = bench(compute_ref, ref_kwargs)
        res_time = bench(
            compute_res,
            dict(res_inputs, symm_mem_handle=handle, symm_mem_buffer=buffer),
        )
        with open(os.path.join(outdir, f"{rank}.json"), "w", encoding="utf8") as f:
            json.dump([ref_time, res_time], f)
    finally:
        dist.destroy_process_group()


@label("v1_megatron_rs_add_rms_ag_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_rs_add_rms_ag_(config):
    import cloudpickle
    world = config[0]
    with tempfile.TemporaryDirectory(prefix="v1-rs-rms-ag-") as tmp:
        mp.start_processes(
            worker,
            args=(config, os.path.join(tmp, "rdzv"), tmp,
                  cloudpickle.dumps(kernelgenbench.triton.v1_megatron_rs_add_rms_ag_)),
            nprocs=world,
            join=True,
            start_method="spawn",
        )
        times = [json.load(open(os.path.join(tmp, f"{rank}.json"), encoding="utf8"))
                 for rank in range(world)]
    ref_time = max(item[0] for item in times)
    res_time = max(item[1] for item in times)
    return CustomBenchmarkResult(
        ref_time=ref_time,
        res_time=res_time,
        speedup=ref_time / res_time if res_time else float("inf"),
    )
