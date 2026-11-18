"""Reference: SGLang CP interleaved-token all-gather (zigzag strategy).

Official implementation (pinned SGLang commit
``62384de1ae5e3bbd3159afd1639525b0bf39fbf9``):

* ``ZigzagCPStrategy._all_gather_reorganized``:
  https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/cp/zigzag.py#L510-L547
* ``ZigzagCPStrategy.gather_hidden_states``:
  https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/cp/zigzag.py#L366-L375

Zigzag CP gives every rank two ends of the sequence instead of one contiguous
block, so restoring the global order needs the split lengths of the reorganized
buffer plus the inverse zigzag index.

Contract:

* each rank contributes ``x`` whose first ``logical_lengths[rank]`` rows are
  real; the function trims to that length and zero-pads to
  ``max(logical_lengths)``;
* a real all-gather of the padded shard over the CP group;
* the gathered rank-major buffer is trimmed per rank back to its logical length
  and concatenated (``_all_gather_reorganized``);
* that buffer is then split by ``split_lengths`` and reordered by
  ``reverse_indices``, which is the inverse zigzag permutation
  (``gather_hidden_states``);
* the input is only read; the result is a fresh contiguous tensor.

The reference performs the same steps with plain tensor ops on the group.
"""
import torch
import torch.distributed as dist


def v1_sgl_cp_zigzag_gather(x, logical_lengths, physical_lengths, total_tokens,
                            group, split_lengths, reverse_indices):
    cp_size = group.size()
    rank = dist.get_rank(group)
    max_len = max(physical_lengths)
    local = logical_lengths[rank]
    assert x.shape[0] >= local, (x.shape[0], local)
    shard = x[:local]
    pad = max_len - shard.shape[0]
    if pad > 0:
        shard = torch.nn.functional.pad(shard, [0, 0] * (shard.ndim - 1) + [0, pad])
    gathered = torch.empty(max_len * cp_size, *shard.shape[1:], dtype=shard.dtype,
                           device=shard.device)
    dist.all_gather_into_tensor(gathered, shard.contiguous(), group=group)
    chunks = torch.split(gathered, [max_len] * cp_size, dim=0)
    reorganized = torch.cat([chunks[each][:length]
                             for each, length in enumerate(logical_lengths)], dim=0)
    pieces = torch.split(reorganized, list(split_lengths), dim=0)
    out = torch.cat([pieces[each] for each in reverse_indices], dim=0)
    return out.contiguous()
