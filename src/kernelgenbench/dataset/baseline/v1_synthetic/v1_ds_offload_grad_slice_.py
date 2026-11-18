"""DeepSpeed ZenFlow gradient-slice cast/copy into the fp32 master buffer.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/zenflow/zenflow_stage_1_and_2.py#L678-L693

Contract:
  * The source is ``gradient.view(-1).narrow(0, source_offset, num_elements)``; a
    view is used, so the gradient tensor itself is never written.
  * A source whose dtype differs from ``master_dtype`` is cast (allocating a new
    tensor) before the copy.
  * The destination is ``destination.view(-1).narrow(0, dest_offset, num_elements)``
    and receives the source with ``copy_(..., non_blocking=True)``; the caller's
    destination tensor is mutated in place, nothing is returned.
  * ``param.grad`` is always cleared to ``None`` (offload only), even when
    ``num_elements`` is 0. The parameter data is untouched.
"""
import torch


def v1_ds_offload_grad_slice_(param, gradient, destination, source_offset, dest_offset,
                              num_elements, master_dtype):
    src = gradient.view(-1).narrow(0, source_offset, num_elements)
    if src.dtype != master_dtype:
        src = src.to(master_dtype)
    destination.view(-1).narrow(0, dest_offset, num_elements).copy_(src, non_blocking=True)
    param.grad = None
    return None
