"""Reference: SGLang DSA indexer top-k broadcast from rank 0 (in place).

Official implementation (pinned SGLang commit
``62384de1ae5e3bbd3159afd1639525b0bf39fbf9``):

* ``_broadcast_indexer_topk_from_rank0_impl``:
  https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L207-L224
* eager wrapper ``_broadcast_indexer_topk_from_rank0``:
  https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L227-L239

Row 551's original reference documented a DeepInfer CP broadcast, but DeepInfer
is not one of the pinned frameworks of this project, so that branch is removed
(see decision.json).  The pinned SGLang tree has the corresponding in-place
broadcast for the DSA indexer's finalized top-k indices, and that is the entry
this reference states.

Contract:

* ``topk_indices`` is an int32 or int64 tensor that is overwritten in place with
  rank 0's values; the function returns ``None``;
* the broadcast runs over the attention-TP group (``src = 0``, group-local);
* with a single rank the function returns immediately without a collective and
  the tensor is unchanged;
* a ``None`` input is passed through by the eager wrapper;
* the group is used as given, so the caller owns process-group setup.
"""
import torch
import torch.distributed as dist


def v1_sgl_dsa_topk_broadcast_(topk_indices, group):
    if topk_indices is None:
        return None
    assert topk_indices.dtype in (torch.int32, torch.int64), topk_indices.dtype
    if group.size() == 1:
        return None
    source = dist.get_global_rank(group, 0)
    dist.broadcast(topk_indices, src=source, group=group)
    return None
