"""DeepSpeed vocabulary-parallel entropy, single-rank specialization.

Official implementation:
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/vocab_parallel_ops.py#L257-L297
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/vocab_parallel_ops.py#L431-L435

This standalone reference keeps the single-shard contract: ``logits`` already
contains the whole vocabulary, so the local row statistics are also the global
ones.  Per-token results keep the leading shape, ``logits.shape[:-1]`` (the
official op returns ``[...,]`` rather than a flattened row vector).  Rows that
are entirely ``-inf`` have no distribution and yield NaN; individual ``-inf``
entries contribute nothing.
"""
import torch


def v1_vocab_entropy(logits):
    x = logits.reshape(-1, logits.shape[-1]).float()
    logp = torch.log_softmax(x, dim=-1)
    terms = (torch.exp(logp) * logp).masked_fill(logp == float('-inf'), 0.0)
    return (-terms.sum(dim=-1)).reshape(logits.shape[:-1])
