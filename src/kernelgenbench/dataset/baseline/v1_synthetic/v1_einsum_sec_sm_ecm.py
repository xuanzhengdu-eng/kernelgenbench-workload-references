"""DeepSpeed MoE einsum helper, ``sec,sm->ecm`` rule (723-row 97).

Source (pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/sharded_moe.py#L117-L149

Official ``deepspeed.moe.sharded_moe.einsum(rule, a, b)`` has two reachable
bodies:
  * ``USE_EINSUM=True`` (module default) forwards every rule to ``torch.einsum``;
  * ``USE_EINSUM=False`` selects a hand-written rewrite per rule; for
    ``sec,sm->ecm`` it is ``matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)``.
Both compute the same contraction but with a different reduction order, so the
low-order bits can differ.  The rewrite cannot reshape an empty ``s`` axis,
which the einsum body accepts; that divergence is recorded, not asserted.
"""
import torch


def v1_einsum_sec_sm_ecm(rule, a, b, use_einsum=True):
    # Only the ``sec,sm->ecm`` rule is covered by this item; the official helper
    # itself accepts any rule, so the reference does not police other rules.
    if use_einsum:
        return torch.einsum(rule, a, b)
    s, e, c = a.shape
    m = b.shape[1]
    return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
