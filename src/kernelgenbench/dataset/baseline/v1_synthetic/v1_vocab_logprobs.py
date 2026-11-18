"""DeepSpeed vocabulary selected log-probability, single vocabulary shard.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/vocab_parallel_ops.py#L199-L229

``_VocabParallelLogProbs.forward`` for one shard (``tp_group=None``,
``vocab_start_index=0``): flatten the leading dimensions, take the fp32 global
max and shifted log-sum-exp, gather the requested ids and zero the entries whose
id equals ``ignore_index``.  The result is ``(rows, len(index))`` fp32.

The legacy reference also carried an SGLang branch.  That branch is the already
registered item ``v1_sgl_spec_logprobs`` (same official
``compute_spec_logprobs`` lines); it is not duplicated here.
"""
import torch

def v1_vocab_logprobs(logits, index, ignore_index):
    x = logits.reshape(-1, logits.shape[-1]).float()
    idx = index.reshape(x.shape[0], index.shape[-1])
    valid = idx != ignore_index
    values = torch.log_softmax(x, dim=-1).gather(1, idx.clamp(0, x.shape[-1] - 1))
    return torch.where(valid, values, 0.0)
