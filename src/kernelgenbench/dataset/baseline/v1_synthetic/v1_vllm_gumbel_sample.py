"""vLLM seeded Murmur3 Gumbel-max sampling without logits-cache writes.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/gumbel.py#L80-L212

Scope: the Murmur3 hash and Gumbel-argmax math of ``gumbel_sample`` with no
``logits_cache``/``logits_cache_col`` arguments, so the draft cache side effects
are out of scope. The selected no-cache adapter still covers the public
``apply_temperature`` and ``is_drafting`` switches and the full fp32 score path:

* ``expanded_idx_mapping`` row ``-1`` is an invalid request: temperature 0 and
  seed 0 are substituted, so that row degenerates to a plain argmax.
* The noise key mixes the request seed (both 32-bit halves), the row position
  (plus 1 << 30 while drafting) and the vocabulary offset through Murmur3.
* ``temperature == 0`` keeps the row noise-free; otherwise the score is
  ``logits / temperature + gumbel_noise``. ``apply_temperature`` only controls
  the division.
* Returns int64 ``[num_tokens]``; inputs are read-only.
"""
import torch


def v1_vllm_gumbel_sample(
    logits,
    expanded_idx_mapping,
    temperature,
    seed,
    pos,
    apply_temperature,
    is_drafting,
):
    assert logits.ndim == 2 and logits.shape[0] > 0 and logits.shape[1] > 0
    assert logits.dtype in (torch.float32, torch.float16, torch.bfloat16)
    tokens, vocab = logits.shape
    assert expanded_idx_mapping.shape == (tokens,) and pos.shape == (tokens,)
    assert temperature.ndim == seed.ndim == 1
    assert temperature.shape == seed.shape and temperature.numel() > 0
    assert expanded_idx_mapping.dtype in (torch.int32, torch.int64)
    assert pos.dtype == seed.dtype == torch.int64
    assert all(
        x.device == logits.device
        for x in (expanded_idx_mapping, temperature, seed, pos)
    )
    assert torch.isfinite(logits).all()
    assert torch.isfinite(temperature).all() and (temperature >= 0).all()
    assert ((expanded_idx_mapping >= -1) & (expanded_idx_mapping < seed.numel())).all()

    mask32 = 0xFFFFFFFF
    req = expanded_idx_mapping.long().clamp_min(0)
    valid = expanded_idx_mapping >= 0
    temp = torch.where(
        valid, temperature[req].float(), torch.zeros(tokens, device=logits.device)
    )
    seeds = seed[req]
    offsets = torch.arange(vocab, device=logits.device, dtype=torch.int64).view(1, vocab)
    positions = pos + (1 << 30 if is_drafting else 0)

    def rot(value, shift):
        return ((value << shift) | (value >> (32 - shift))) & mask32

    def mix(h, key):
        key = (key * 0xCC9E2D51) & mask32
        key = rot(key, 15)
        key = (key * 0x1B873593) & mask32
        h = rot(h ^ key, 13)
        return (h * 5 + 0xE6546B64) & mask32

    h = torch.zeros((tokens, 1), device=logits.device, dtype=torch.int64)
    for key in (
        seeds.view(tokens, 1) & mask32,
        (seeds.view(tokens, 1) >> 32) & mask32,
        positions.view(tokens, 1) & mask32,
        offsets,
    ):
        h = mix(h, key)
    h = h ^ 16
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & mask32
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & mask32
    h = (h ^ (h >> 16)) & mask32

    # Split the 32 random bits into two exact fp32 halves.
    u = ((h >> 16).float() * (2.0**-16)) + (((h & 0xFFFF).float() + 0.5) * (2.0**-32))
    # Degree-8 log1p series, used for u < 0.25 where the winning tail lives.
    p = torch.full_like(u, 1.0 / 8.0)
    for coefficient in (1.0 / 7, 1.0 / 6, 1.0 / 5, 1.0 / 4, 1.0 / 3, 1.0 / 2, 1.0):
        p = coefficient + u * p
    log1p = torch.where(u < 0.25, -u * p, torch.log((1.0 - u).clamp_min(2.0**-24)))
    noise = -torch.log(-log1p)

    base = logits.float()
    if apply_temperature:
        base = base / torch.where(temp == 0, torch.ones_like(temp), temp).view(tokens, 1)
    score = base + torch.where((temp != 0).view(tokens, 1), noise, torch.zeros_like(noise))
    return torch.argmax(score, dim=-1).to(torch.int64)
