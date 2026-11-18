"""DeepSpeed saved-softmax vocabulary-parallel cross-entropy backward.

Official implementation (saved-softmax backward):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/cross_entropy.py#L95-L105
The saved tensors are produced by the forward pass:
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/cross_entropy.py#L64-L92

Contract: ``softmax`` is the local vocabulary shard of an FP32 softmax,
``target_mask`` marks rows whose target is outside this shard (or ignored),
``masked_target`` is the in-shard target index, and ``valid_target`` marks
rows that contribute at all.  The DeepSpeed path returns a new tensor; it does
not mutate ``softmax``.  The legacy Megatron-fork explicit-backward variants
(``megatron``/``megatron_fused``) have no standalone NVIDIA Megatron entry at
the pinned commit and are not registered.
"""
import torch


def v1_megatron_vocab_parallel_ce_backward_deepspeed(softmax, target_mask, masked_target,
                                            grad_output, label_smoothing,
                                            backend='deepspeed', valid_target=None,
                                            logits_dtype=None):
    if backend != 'deepspeed':
        raise NotImplementedError(
            'Legacy Megatron-fork explicit CE backward has no official NVIDIA '
            'Megatron entry at 266f1c97; only the DeepSpeed saved-softmax path is kept')
    assert label_smoothing == 0 and valid_target is not None and logits_dtype is not None
    n = softmax.shape[-1]
    result = softmax.clone()
    rows = torch.arange(masked_target.numel(), device=softmax.device)
    result.view(-1, n)[rows, masked_target] -= (~target_mask.view(-1)).float()
    return (result * valid_target.unsqueeze(-1) * grad_output.unsqueeze(-1)).to(logits_dtype)
