"""vLLM CUDA indexer-K quantize + paged cache store (_C_cache_ops).

Pinned source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L3536-L3545
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/cache_kernels.cu#L2069-L2104
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/cache_kernels.cu#L706-L773

k is bf16 (tokens, 128) and kv_cache is uint8 (blocks, block_size, 128 + 4).  The value
area is always row-major; the trailing 4 bytes per slot hold one fp32 scale, a UE8M0
exponent, or four packed MXFP8 exponent bytes (scale_fmt='mxfp8', which also requires
head_dim == quant_block_size == 128).  Negative slots are skipped, the cache is mutated
in place and the call returns None.  The ROCm tiled-layout variant has a different
value layout and lives in v1_vllm_indexer_quant_store_.py.
"""
import torch


def v1_vllm_indexer_quant_store_cuda(
    k, kv_cache, slot_mapping, quant_block_size, scale_fmt
):
    assert k.dtype == torch.bfloat16 and k.shape[-1] == 128
    assert quant_block_size == 128 and scale_fmt in ("fp32", "ue8m0", "mxfp8")
    num_blocks, block_size, _ = kv_cache.shape
    head_dim = k.shape[-1]
    raw = kv_cache.view(num_blocks, -1)
    for i in range(slot_mapping.numel()):
        slot = int(slot_mapping[i])
        if slot < 0:
            continue
        page, offset = divmod(slot, block_size)
        x = k[i].float()
        scale = x.abs().max().clamp_min(1e-4) / 448.0
        if scale_fmt in ("ue8m0", "mxfp8"):
            scale = torch.exp2(torch.ceil(torch.log2(scale)))
        quantized = (x / scale).to(torch.float8_e4m3fn).view(torch.uint8)
        raw[page, offset * head_dim : (offset + 1) * head_dim] = quantized
        if scale_fmt == "mxfp8":
            exponent = (int(scale.view(torch.int32).item()) >> 23) & 255
            raw[page, block_size * head_dim + offset * 4 : block_size * head_dim + (offset + 1) * 4] = exponent
        else:
            raw[page, block_size * head_dim :].view(torch.float32)[offset] = scale
    return None
