"""SGLang DCP cached-prefix gather (dcp_cache mode).

Official source (pinned commit):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/dcp/comm.py#L337-L437
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/dcp/comm.py#L162-L183

``all_gather_kv_cache_for_dcp`` packs ``kv_a`` and ``k_pe`` along the last
dimension, pads every request to the rank-aligned window, all-gathers the packed
rows, and re-orders them to request/position order.  ``prefix_starts_cpu``
offsets each request inside the DCP round-robin cycle and ``interleave_size``
groups that many consecutive positions per rank.  With DCP disabled it returns
the packed tensor unchanged.  The reference keeps the same padding/cursor
arithmetic in pure torch and uses the caller's process group for the real
collective.
"""
import torch
import torch.distributed as dist


def v1_sgl_dcp_gather_cache_(local_rows, k_pe, seq_lens_cpu, group=None,
                             prefix_starts_cpu=None, interleave_size=1,
                             dcp_enabled=True):
    assert k_pe is not None
    assert local_rows.shape[:-1] == k_pe.shape[:-1]
    assert interleave_size > 0
    packed = torch.cat((local_rows, k_pe), dim=-1)
    if not dcp_enabled:
        return packed
    world = dist.get_world_size(group)
    lens = seq_lens_cpu.tolist()
    starts = prefix_starts_cpu.tolist() if prefix_starts_cpu is not None else [0] * len(lens)
    totals = [sum(1 for n, start in zip(lens, starts) for j in range(n)
                  if ((start + j) // interleave_size) % world == rank)
              for rank in range(world)]
    pad = packed.new_zeros((max(totals),) + packed.shape[1:])
    pad[:packed.shape[0]].copy_(packed)
    shards = [torch.empty_like(pad) for _ in range(world)]
    dist.all_gather(shards, pad, group=group)
    cursors = [0] * world
    out = []
    for n, start in zip(lens, starts):
        for j in range(n):
            rank = ((start + j) // interleave_size) % world
            out.append(shards[rank][cursors[rank]])
            cursors[rank] += 1
    if not out:
        return packed.new_empty((0,) + packed.shape[1:])
    return torch.stack(out)
