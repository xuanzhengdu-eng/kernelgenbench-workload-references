"""vLLM batched pointer/offset KV copy (swap_blocks_batch mode).

Official entry: torch.ops._C_cache_ops.swap_blocks_batch in
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/cache_kernels.cu#L93-L230
Python wrapper: vllm._custom_ops.swap_blocks_batch

The native interface takes CPU int64 arrays of raw source pointers, destination
pointers and byte sizes (cuMemcpyBatchAsync on CUDA >= 12.8, otherwise a
cudaMemcpyAsync loop).  This reference presents the legacy descriptor adapter:
src/dst are lists of 1-D uint8 CUDA buffers and block_mapping is a CPU int64
[N,5] tensor of (src_buffer, src_offset, dst_buffer, dst_offset, nbytes).
Ranges of one descriptor must fit in their buffers; destination ranges must not
overlap each other.  Source buffers are unchanged; returns None.
"""
import torch


def v1_vllm_batch_kv_copy(src, dst, block_mapping):
    assert block_mapping.device.type == "cpu" and block_mapping.dtype == torch.int64
    assert block_mapping.ndim == 2 and block_mapping.shape[1] == 5
    for si, so, di, do, n in block_mapping.tolist():
        assert n > 0 and 0 <= si < len(src) and 0 <= di < len(dst)
        assert so >= 0 and do >= 0 and so + n <= src[si].numel() and do + n <= dst[di].numel()
        dst[di][do:do + n].copy_(src[si][so:so + n])
    return None
