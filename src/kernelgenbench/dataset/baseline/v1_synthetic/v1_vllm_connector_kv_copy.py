"""vLLM per-layer KV block copy between host and device (connector mode).

Official entry: vllm.distributed.kv_transfer.kv_connector.utils.copy_kv_blocks in
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/kv_transfer/kv_connector/utils.py#L235-L267

src/dst are dicts of per-layer KV tensors and the two block-id lists are paired
(h2d: source host blocks -> destination device blocks; d2h: the reverse).
Empty dicts/lists or mismatched lengths are a no-op.  Destination tensors are
mutated; sources are read-only; returns None.  Indices are made device-local by
the upstream helper; this reference expresses the same observable copy.
"""
import torch


def v1_vllm_connector_kv_copy(src, dst, src_block_ids, dst_block_ids, direction):
    if not src or not dst or not src_block_ids or not dst_block_ids or len(src_block_ids) != len(dst_block_ids):
        return None
    if direction not in ("h2d", "d2h"):
        raise ValueError("unsupported direction")
    for name in src:
        # Snapshot the selected blocks before any destination writes.
        selected = src[name][src_block_ids].to(dst[name].device)
        dst[name][dst_block_ids] = selected
    return None
