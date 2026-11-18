"""DeepSpeed ZeRO-3 gradient partition copy/accumulation and release.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/zenflow/engine_stage3.py#L457-L527

Contract (single data-parallel group, on-accelerator buffers, no optimizer offload):
  * A parameter whose slice holds no real data,
    ``partition_numel() * rank >= ds_numel``, only has ``grad`` released and is
    skipped: it neither contributes a returned buffer nor touches its grad buffer.
    With rank 0 that is exactly the ``ds_numel == 0`` case.
  * Otherwise the target is ``grad_buffers[ds_id].narrow(0, 0, grad.numel())`` and
    is returned in order, so the result aliases the persistent grad buffer.
  * ``micro_step == 0`` overwrites the target with the partition (non-blocking);
    later micro steps add ``grad.to(accum_dtype).view(target.shape)`` in place.
  * Every visited parameter has ``param.grad`` set to ``None``; the incoming
    ``grad_partitions`` are never modified.
"""
import torch


def v1_ds_partition_grads_(params, grad_partitions, grad_buffers, rank, micro_step,
                           accum_dtype):
    buffers = []
    for param, grad_partition in zip(params, grad_partitions):
        if param.partition_numel() * rank >= param.ds_numel:
            param.grad = None
            continue
        target = grad_buffers[param.ds_id].narrow(0, 0, grad_partition.numel())
        buffers.append(target)
        if micro_step == 0:
            target.copy_(grad_partition, non_blocking=True)
        else:
            target.add_(grad_partition.to(accum_dtype).view(target.shape))
        param.grad = None
    return buffers
