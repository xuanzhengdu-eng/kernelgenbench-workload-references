"""vLLM TurboQuant AoS store: FP8 or MSE key and 3/4-bit uniform V.

Official pinned vLLM AoS module: shared value quantizer #L25-L137, FP8-key kernel
#L145-L212, MSE-key kernel #L221-L343, launcher #L350-L447:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/ops/triton_turboquant_store.py#L25-L447

Caller-visible contract: ``triton_turboquant_store`` writes one uint8 cache row
per (token, head) at ``slot_mapping[t]`` and returns None.  ``key_fp8=True``
casts the raw key to E4M3 at bytes [0, D); ``key_fp8=False`` normalises the key,
rotates it with ``PiT``, bucketizes it against ``midpoints`` into 3/4-bit
indices and stores the fp16 norm after the packed indices.  The value is always
uniformly quantized to ``value_quant_bits`` bits with an fp16 (scale, min) pair
appended.  Slots < 0 are padding and are not written.
"""
import torch


def _pack(q, bits):
    if bits == 4:
        return (q[::2] | (q[1::2] << 4)).to(torch.uint8)
    q = q.reshape(-1, 8)
    w = (q << (torch.arange(8, device=q.device) * 3)).sum(-1)
    return torch.stack((w & 255, (w >> 8) & 255, (w >> 16) & 255), -1).flatten().to(torch.uint8)


def _fp8(key, value, kv_cache, slot_mapping, PiT, midpoints, mse_bits,
         key_packed_size, value_quant_bits, key_fp8=True):
    if not key_fp8 or value_quant_bits not in (3, 4):
        raise ValueError('FP8 K and 3/4-bit V only')
    n, h, d = key.shape
    bs = kv_cache.shape[1]
    bits = value_quant_bits
    vbytes = d * bits // 8
    for t in range(n):
        slot = int(slot_mapping[t])
        if slot < 0:
            continue
        for j in range(h):
            v = value[t, j].float()
            vmin = v.min()
            scale = torch.clamp((v.max() - vmin) / (2 ** bits - 1), min=1e-8)
            q = ((v - vmin) / scale + .5).to(torch.int32).clamp(0, 2 ** bits - 1)
            if bits == 4:
                packed = (q[::2] | (q[1::2] << 4)).to(torch.uint8)
            else:
                q = q.reshape(-1, 8)
                shifts = torch.arange(8, device=q.device) * 3
                w = (q << shifts).sum(-1)
                packed = torch.stack((w & 255, (w >> 8) & 255, (w >> 16) & 255),
                                     -1).flatten().to(torch.uint8)
            raw = kv_cache[slot // bs, slot % bs, j]
            raw[:d].copy_(key[t, j].to(torch.float8_e4m3fn).view(torch.uint8))
            raw[key_packed_size:key_packed_size + vbytes].copy_(packed)
            raw[key_packed_size + vbytes:key_packed_size + vbytes + 4].copy_(
                torch.stack((scale, vmin)).to(torch.float16).view(torch.uint8))


def _mse(key, value, kv_cache, slot_mapping, PiT, midpoints, mse_bits,
         key_packed_size, value_quant_bits, key_fp8=False):
    if key_fp8 or mse_bits not in (3, 4) or value_quant_bits not in (3, 4):
        raise ValueError('MSE key and 3/4-bit K/V only')
    n, h, d = key.shape
    bs = kv_cache.shape[1]
    kbytes = d * mse_bits // 8
    vbytes = d * value_quant_bits // 8
    norm = key.float().reshape(-1, d).norm(dim=1, keepdim=True)
    y = (key.float().reshape(-1, d) / (norm + 1e-8)) @ PiT
    for t in range(n):
        slot = int(slot_mapping[t])
        if slot < 0:
            continue
        for j in range(h):
            raw = kv_cache[slot // bs, slot % bs, j]
            kq = torch.bucketize(y[t * h + j].contiguous(), midpoints,
                                 right=True).to(torch.int32)
            raw[:kbytes].copy_(_pack(kq, mse_bits))
            raw[kbytes:kbytes + 2].copy_(norm[t * h + j].to(torch.float16).view(torch.uint8))
            v = value[t, j].float()
            vmin = v.min()
            scale = torch.clamp((v.max() - vmin) / (2 ** value_quant_bits - 1), min=1e-8)
            q = ((v - vmin) / scale + .5).to(torch.int32).clamp(0, 2 ** value_quant_bits - 1)
            raw[key_packed_size:key_packed_size + vbytes].copy_(
                _pack(q, value_quant_bits))
            raw[key_packed_size + vbytes:key_packed_size + vbytes + 4].copy_(
                torch.stack((scale, vmin)).to(torch.float16).view(torch.uint8))


def v1_vllm_turboquant_store_(key, value, kv_cache, slot_mapping, PiT, midpoints,
                              mse_bits, key_packed_size, value_quant_bits,
                              key_fp8=False):
    fn = _fp8 if key_fp8 else _mse
    return fn(key, value, kv_cache, slot_mapping, PiT, midpoints, mse_bits,
              key_packed_size, value_quant_bits, key_fp8)
