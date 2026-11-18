"""Megatron mixed precision gradient unscale and nonfinite check.

https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/optimizer/optimizer.py#L497-L522

This is inside ``MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan``:
reset ``found_inf``, run the fused
``torch._amp_foreach_non_finite_check_and_unscale_`` over the main gradients,
MAX-reduce ``found_inf`` over the grad-stats group, and return whether anything
was nonfinite.  A stub optimizer skips collection and unscaling but still resets
and reduces.  The legacy fork reference looped over a *list* of groups; the
official method takes the single group returned by
``get_grad_stats_parallel_group()``, so this item is scoped to one group.
"""
import torch
import torch.distributed as dist

def v1_megatron_unscale_check_(grads, found_inf, inv_scale, group, is_stub_optimizer=False):
    found_inf.fill_(0.0)
    if not is_stub_optimizer:
        torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
    dist.all_reduce(found_inf, op=dist.ReduceOp.MAX, group=group)
    return bool(found_inf.item() > 0)
