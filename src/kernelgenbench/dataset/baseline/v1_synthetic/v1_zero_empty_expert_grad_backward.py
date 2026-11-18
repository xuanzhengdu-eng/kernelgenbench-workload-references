"""DeepSpeed empty-expert gradient masking, official backward.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_experts.py#L269-L290

``_ZeroEmptyExpertGrad.backward`` returns ``(None, None)`` for a None gradient,
returns the *same* grad object when every expert is active, and otherwise zeroes
only the inactive expert slices.  A non-contiguous grad cannot be written in
place, so it is made contiguous first and the copy is the returned tensor; the
caller's original buffer keeps its values.  ``num_tokens_per_expert`` is only
read.
"""
import torch

def v1_zero_empty_expert_grad_backward(grad, num_tokens_per_expert):
    if grad is None:
        return None, None
    inactive = num_tokens_per_expert == 0
    # Every expert is live on the common path: return the buffer the grouped GEMM produced.
    if not bool(inactive.any()):
        return grad, None
    # Zero only the empty-expert rows; a non-contiguous grad cannot be written in place.
    if not grad.is_contiguous():
        grad = grad.contiguous()
    grad[inactive] = 0
    return grad, None
