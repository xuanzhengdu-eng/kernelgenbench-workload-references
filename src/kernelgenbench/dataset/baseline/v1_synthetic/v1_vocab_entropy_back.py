"""DeepSpeed vocabulary entropy backward, official saved statistics.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/vocab_parallel_ops.py#L279-L299

``_VocabParallelEntropy.backward`` for one vocabulary shard:
``dH/dz_j = -grad_output * p_j * (log p_j + H)`` with ``p`` reconstructed from
the saved fp32 global max and shifted log-sum-exp.  Terms whose log-probability
is exactly ``-inf`` are masked to zero (so ``0 * -inf`` cannot poison a row), and
a row whose global max is ``-inf`` (no softmax) gets an all-zero gradient.  The
output keeps the logits' shape and dtype.
"""
import torch

def v1_vocab_entropy_back(logits, global_max, log_sum_exp, entropy, grad_output):
    x = logits.reshape(-1, logits.shape[-1]).float()
    logp = x - global_max[:, None] - log_sum_exp[:, None]
    grad = (torch.exp(logp) * (logp + entropy[:, None])).masked_fill(logp == float("-inf"), 0.0)
    grad = (-grad * grad_output.reshape(-1, 1).float()).masked_fill(
        (global_max == float("-inf"))[:, None], 0.0)
    return grad.reshape_as(logits).to(logits.dtype)
