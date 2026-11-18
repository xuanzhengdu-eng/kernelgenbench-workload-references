"""Reference: DeepSpeed equal-split all-to-all for MoE dispatch.

Official implementation (pinned DeepSpeed commit
``8af59b2e7a91c227081617240e98aa282eadc5f9``):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/sharded_moe.py#L97-L109

DeepSpeed's ``_AllToAll`` is the equal-split all-to-all used by
``TopKGate``/``MOELayer`` to exchange tokens between expert-parallel ranks:

* ``forward(ctx, group, input)`` — one argument, no split lists; the output is
  always a fresh ``empty_like`` tensor, even for a single rank (so the result
  never aliases the input);
* ``backward`` returns a two-element tuple ``(None, grad)`` and applies the same
  all-to-all to the incoming gradient (its own gradient is the reverse
  exchange);
* the input is made contiguous and only read.

Contract: every rank must contribute the same shape (equal split); the number
of rows must be divisible by the group size for a meaningful exchange.
"""
import torch
import torch.distributed as dist


def v1_deepspeed_all_to_all(group, input_):
    assert group is not None
    local = input_.contiguous()
    out = torch.empty_like(local)
    dist.all_to_all_single(out, local, group=group)
    return out
