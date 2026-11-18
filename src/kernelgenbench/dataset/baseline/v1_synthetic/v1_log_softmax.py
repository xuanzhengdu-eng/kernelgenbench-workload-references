# Source-checked synthetic-workload benchmark reference.
"""OP-da759446a0ee: Megatron Triton log_softmax, last dimension only, no mask.

Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/custom_layers/batch_invariant_kernels.py#L308-L340

Contract kept from the old reference:
  * only ``dim == -1`` or the last dimension is accepted; anything else raises
    ``ValueError`` (the official wrapper raises the same error).
  * the row maximum and the ``log(sum(exp))`` accumulation run in fp32 on every
    dtype, and the result is returned in the input dtype.
  * arbitrary rank: the official wrapper flattens everything but the last axis
    into rows, so leading dims are unrestricted.
  * returns a fresh tensor and never modifies the input.
"""
import torch


def v1_log_softmax(input, dim):
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError("only last dimension is supported")
    return torch.log_softmax(input.float(), dim=-1).to(input.dtype)
