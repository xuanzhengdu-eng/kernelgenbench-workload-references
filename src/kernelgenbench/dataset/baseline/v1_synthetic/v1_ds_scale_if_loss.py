"""DeepSpeed loss scaling, official ``ZeROOptimizer.scale_if_loss`` contract.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/base_optimizer.py#L428-L441
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/utils.py#L1332-L1342
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/fp16/loss_scaler.py#L165-L169

``scale_if_loss`` only touches values that are loss tensors: a tensor with
exactly one element and a non-None ``grad_fn``.  For those:
* ``mode='custom'`` -> ``external_loss_scale * value``;
* ``mode='internal'`` -> ``value * loss_scaler.loss_scale``;
* otherwise, and for every non-loss value, the original object is returned.

The reference's three modes represent custom scaler availability, an internal
DeepSpeed scaler, and no scaler.  A torch ``GradScaler`` is not covered because
the pinned upstream branch handles it through a separate object with its own
``scale`` method, which this synthetic entry does not model.
"""
import torch


def v1_ds_scale_if_loss(value, scale, mode):
    assert mode in ("custom", "internal", "none")
    if isinstance(value, torch.Tensor) and value.numel() == 1 and value.grad_fn is not None:
        if mode == "custom":
            return scale * value
        if mode == "internal":
            return value * scale
    return value
