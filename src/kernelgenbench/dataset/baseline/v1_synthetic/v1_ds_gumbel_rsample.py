"""DeepSpeed unit Gumbel sampling.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/sharded_moe.py#L79-L86

Returns fresh standard Gumbel samples with the current PyTorch device RNG.
DeepSpeed caches the distribution per device; this scoped reference assumes the
default float32 dtype throughout, so reuse does not change sample semantics.
Supported shapes include scalars ``()`` and empty shapes.
"""
import torch


def v1_ds_gumbel_rsample(shape, device):
    zero = torch.tensor(0.0, device=device)
    one = torch.tensor(1.0, device=device)
    return torch.distributions.gumbel.Gumbel(zero, one).rsample(shape)
