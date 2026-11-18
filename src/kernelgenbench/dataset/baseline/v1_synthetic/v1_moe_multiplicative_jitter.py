"""DeepSpeed multiplicative MoE input jitter with the module-global sampler cache.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/sharded_moe.py#L55-L77

Contract:
  * ``epsilon == 0`` returns the input object itself, without consuming RNG and
    without creating or touching the cache.
  * The first non-zero epsilon for a device builds
    ``Uniform(1 - epsilon, 1 + epsilon).rsample`` once and stores it in the
    module-global ``uniform_map`` under that device. Later calls on the same device
    reuse that sampler, so a changed epsilon only changes the sampled shape, not
    the bounds; the cache keeps at most one entry per device.
  * The sampler draws from the caller's global RNG stream, so the random state is
    part of the observable behavior; ``x`` itself is never modified.
  * The result has the shape and dtype of ``x``. Non-floating dtypes are not
    converted: the multiply promotes them (int tensor times float sampler).
  * There is no epsilon validation beyond what ``torch.distributions.Uniform``
    itself rejects (a negative epsilon gives low > high and raises ValueError).
"""
import torch

uniform_map = {}


def v1_moe_multiplicative_jitter(x, device, epsilon=1e-2):
    if epsilon == 0:
        return x
    sampler = uniform_map.get(device)
    if sampler is None:
        sampler = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon, device=device),
            high=torch.tensor(1.0 + epsilon, device=device),
        ).rsample
        uniform_map[device] = sampler
    return x * sampler(x.shape)
