"""DeepSpeed vocabulary-parallel cross entropy, forward loss (one TP shard).

Official implementation:
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/cross_entropy.py#L64-L92
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/cross_entropy.py#L198-L273

Each rank owns one contiguous vocabulary shard.  The reference performs the
global MAX and SUM reductions explicitly, gathers the target logit only on the
owning shard, returns zero for ``ignore_index`` rows and always returns FP32
without mutating the input logits.  Unlike the Megatron entries this contract
has no label smoothing, takes ``ignore_index`` and never writes back to the
logits, so it is a separate Item.
"""
import torch
import torch.distributed as dist


def v1_megatron_vocab_parallel_ce_deepspeed(logits, target, label_smoothing, rank,
                                            world_size, group, backend='deepspeed',
                                            ignore_index=-100):
    if backend != 'deepspeed':
        raise ValueError('this reference models the DeepSpeed shard contract only')
    assert label_smoothing == 0
    x = logits.float()
    n = x.shape[-1]
    maximum = x.amax(-1)
    if world_size > 1:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    denominator = (x - maximum.unsqueeze(-1)).exp().sum(-1)
    if world_size > 1:
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=group)
    valid = target != ignore_index
    inside = valid & (target >= rank * n) & (target < (rank + 1) * n)
    indices = (target - rank * n).clamp(0, n - 1)
    selected = torch.where(inside, x.gather(-1, indices.unsqueeze(-1)).squeeze(-1),
                           torch.zeros_like(maximum))
    if world_size > 1:
        dist.all_reduce(selected, op=dist.ReduceOp.SUM, group=group)
    return torch.where(valid, denominator.log() + maximum - selected,
                       torch.zeros_like(maximum))
