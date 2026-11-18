"""Reference: Megatron rank-wise all-to-all (equal and unequal first-dim splits).

Official implementation (pinned Megatron-LM commit
``266f1c97dca477ca8b92c16087506da2000b0b84``):

* public wrapper this reference mirrors, ``all_to_all(group, input_,
  output_split_sizes_=None, input_split_sizes=None)``:
  https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L535-L538
* implementation it calls, ``_AllToAll.forward`` (class at L420, forward body
  L422-L451):
  https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L420-L451
* the unequal-split allocation inside that same body (``torch.cuda.current_device()``):
  https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L438-L443

``all_to_all`` is the only public entry to ``_AllToAll``, the autograd function
used by tensor-parallel layers; the wrapper passes ``output_split_sizes_``
through as ``output_split_sizes``:

* ``world_size == 1`` returns the input object itself (an autograd view sharing
  its storage, no collective);
* otherwise the input is made contiguous and a new output is allocated
  (``empty_like`` for the equal split, ``sum(output_split_sizes)`` rows for the
  unequal split), then ``torch.distributed.all_to_all_single`` is issued with
  the two split lists.  The result is a fresh contiguous tensor;
* ``backward`` re-applies the same all-to-all with the split lists swapped, so
  the gradient of an all-to-all is the reverse all-to-all.

Contract: ``output_split_sizes[r]`` is how many rows rank ``r`` sends to this
rank and ``input_split_sizes[r]`` is how many rows this rank sends to rank
``r``; both must be ``None`` together or given together.  Both modes are the
same official entry and the same implementation, selected by whether the split
lists are ``None``, so this file is the single reference for the item.  The
input is only read.
"""
import torch
import torch.distributed as dist


def v1_megatron_all_to_all(group, input_, output_split_sizes=None,
                           input_split_sizes=None):
    assert group is not None
    if group.size() == 1:
        return input_
    local = input_.contiguous()
    if output_split_sizes is None:
        out = torch.empty_like(local)
    else:
        out = local.new_empty((sum(output_split_sizes), *local.shape[1:]),
                              device=torch.cuda.current_device())
    dist.all_to_all_single(out, local, output_split_sizes=output_split_sizes,
                           input_split_sizes=input_split_sizes, group=group)
    return out
