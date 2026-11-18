# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/distributed/parallel_state.py#L1254-L1311
"""SGLang variable-size reduce-scatter reference (official fixed commit).

This replaces the fork-only Megatron ``variable_collectives`` modes that had no
official source in the pinned Megatron commit.  The surviving contract is
SGLang ``GroupCoordinator.reduce_scatterv``
(``python/sglang/srt/distributed/parallel_state.py#L1254-L1311``):

* ``input_`` is a 2-D tensor whose rows are the concatenation of every rank's
  chunk; ``sizes`` is the per-rank row count (``None`` means equal division);
* the reduction is a sum over ranks (no averaging), and this rank returns the
  ``sizes[rank]``-row slice at the prefix-sum offset;
* ``output`` may be supplied (shape checked and written in place) or ``None``
  (a fresh tensor is allocated and returned);
* ``input_`` is read-only; the sum is accumulated in a private clone.

The Solution binds the real pinned SGLang method; the test runs the
deterministic branch (``SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1``), which uses
``GroupCoordinator.all_reduce`` and is the branch available on this host.
"""
import torch
import torch.distributed as dist


def v1_sgl_reduce_scatterv(input_, output=None, sizes=None, group=None):
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    rank = dist.get_rank(group) if dist.is_initialized() else 0
    if sizes is not None:
        assert len(sizes) == world, "sizes must have one entry per rank"
        assert input_.shape[0] == sum(sizes), "input rows must equal sum(sizes)"
        chunk = sizes[rank]
        offset = sum(sizes[:rank])
    else:
        assert input_.shape[0] % world == 0, "rows must divide evenly without sizes"
        chunk = input_.shape[0] // world
        offset = chunk * rank
    output_shape = (chunk,) + tuple(input_.shape[1:])
    if output is None:
        output = torch.empty(output_shape, dtype=input_.dtype, device=input_.device)
    else:
        assert tuple(output.shape) == output_shape, "supplied output has the wrong shape"
    reduced = input_.clone()
    if dist.is_initialized():
        dist.all_reduce(reduced, group=group)
    output.copy_(reduced.narrow(0, offset, chunk))
    return output
