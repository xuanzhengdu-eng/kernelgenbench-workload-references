"""Reference: Megatron all-to-all explicit backward (reverse split lists).

Official implementation (pinned Megatron-LM commit
``266f1c97dca477ca8b92c16087506da2000b0b84``):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L453-L461

``_AllToAll.backward`` (mappings.py, ``def`` at L454, closing paren at L461)
returns a four-element tuple because the autograd function takes four inputs
``(group, input, output_split_sizes, input_split_sizes)``; only the input
gradient is real and the three metadata slots are ``None``.  The gradient of an
all-to-all with ``(output_split_sizes, input_split_sizes)`` is the all-to-all
with the two lists swapped:

``grad_input = all_to_all(group, grad_output, input_split_sizes, output_split_sizes)``

With one rank the forward returned a view of the input, so the gradient passes
through unchanged; the reference still returns the four-element structure of the
official backward.  The metadata arguments are not
differentiable and the group is opaque, hence the ``None`` slots.
"""
import torch
import torch.distributed as dist


def v1_megatron_all_to_all_backward(group, grad_output, output_split_sizes=None,
                                    input_split_sizes=None):
    assert group is not None
    if group.size() == 1:
        grad = grad_output
    else:
        local = grad_output.contiguous()
        if input_split_sizes is None:
            grad = torch.empty_like(local)
        else:
            grad = local.new_empty((sum(input_split_sizes), *local.shape[1:]))
        dist.all_to_all_single(grad, local, output_split_sizes=input_split_sizes,
                               input_split_sizes=output_split_sizes, group=group)
    return (None, grad, None, None)
