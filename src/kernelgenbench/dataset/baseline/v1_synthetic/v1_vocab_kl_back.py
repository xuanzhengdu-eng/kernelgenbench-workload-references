"""DeepSpeed vocabulary-parallel KL backward from the saved softmax statistics.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/vocab_parallel_ops.py#L337-L383

Contract:
  * ``student``/``teacher`` have shape ``(..., vocab)`` and are reshaped to rows;
    ``global_max`` and ``log_sum_exp`` are the ``(2, rows)`` fp32 statistics saved
    by the forward pass (index 0 student, index 1 teacher); ``kl`` is the fp32
    per-row KL of shape ``(rows,)``; ``grad_output`` is ``(rows,)``.
  * With ``reverse=False`` the weighting distribution A is the teacher and the
    target distribution B the student (``KL(teacher || student)``); with
    ``reverse=True`` they swap. Stage-1 gradients are
    ``p_A * (log p_A - log p_B - KL) * grad_output`` and
    ``(p_B - p_A) * grad_output``, with ``0 * -inf`` masked to zero.
  * A row with an all ``-inf`` side has undefined KL: both gradients are zeroed
    instead of propagating NaN.
  * ``need_student``/``need_teacher`` select which gradient is materialized; the
    other is returned as ``None``. Gradients have the input shape and dtype.
"""
import torch


def v1_vocab_kl_back(student, teacher, global_max, log_sum_exp, kl, grad_output,
                     reverse, need_student, need_teacher):
    rows = student.shape[:-1].numel()
    s = student.reshape(rows, student.shape[-1]).float() - global_max[0, :, None] \
        - log_sum_exp[0, :, None]
    t = teacher.reshape(rows, teacher.shape[-1]).float() - global_max[1, :, None] \
        - log_sum_exp[1, :, None]
    a, b = (s, t) if reverse else (t, s)
    pa = a.exp()
    undefined = (global_max == float('-inf')).any(dim=0)[:, None]
    ga = (pa * (a - b - kl[:, None])).masked_fill(a == float('-inf'), 0.0) \
        * grad_output.reshape(-1, 1)
    gb = (b.exp() - pa) * grad_output.reshape(-1, 1)
    ga = ga.masked_fill(undefined, 0.0)
    gb = gb.masked_fill(undefined, 0.0)
    gs, gt = (ga, gb) if reverse else (gb, ga)
    return (gs.reshape_as(student).to(student.dtype) if need_student else None,
            gt.reshape_as(teacher).to(teacher.dtype) if need_teacher else None)
