"""DeepSpeed vocabulary-parallel KL divergence between student and teacher logits.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/vocab_parallel_ops.py#L306-L335

Contract (single vocabulary shard, tp_group=None):
  * ``student``/``teacher`` have the same shape ``(..., vocab)``; they are
    flattened to rows and upcast to fp32. Per row ``global_max`` is the max logit
    and ``log_sum_exp = log(sum(exp(logit - global_max)))`` on each side.
  * ``reverse=False`` computes ``KL(teacher || student)``; ``reverse=True``
    computes ``KL(student || teacher)``. The weighting distribution is the first
    argument of the pair.
  * Terms whose first-distribution log-probability is ``-inf`` contribute zero.
    A token masked to ``-inf`` in the second distribution while finite in the
    first makes the row ``+inf``. A row that is entirely ``-inf`` on either side
    is ``NaN``.
  * The return has shape ``student.shape[:-1]`` (the vocabulary axis removed), in
    fp32, and is a fresh tensor that does not alias the saved KL.
"""
import torch


def v1_vocab_kl(student, teacher, reverse=False):
    s = torch.log_softmax(student.reshape(-1, student.shape[-1]).float(), dim=-1)
    t = torch.log_softmax(teacher.reshape(-1, teacher.shape[-1]).float(), dim=-1)
    a, b = (s, t) if reverse else (t, s)
    terms = (a.exp() * (a - b)).masked_fill(a == float('-inf'), 0.0)
    terms = terms.masked_fill((b == float('-inf')) & (a != float('-inf')), float('inf'))
    return terms.sum(dim=-1).view(student.shape[:-1])
