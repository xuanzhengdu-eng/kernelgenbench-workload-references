"""vLLM repetition penalty, torch/CUDA routes, in place.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L382-L395

``apply_repetition_penalties`` dispatches to the CUDA op when ``logits`` is a
contiguous CUDA tensor and to the torch route otherwise (for example strided
rows). Both routes share one contract:

* ``logits`` is ``[num_seqs, vocab]``; masks are bool with the same shape and
  ``repetition_penalties`` is ``[num_seqs]`` in the logits dtype.
* A token in the prompt or the output is scaled: positive logits divide by the
  penalty, zero and negative logits multiply by it. Other tokens are untouched.
* Mutates ``logits`` and returns ``None``.
"""
import torch


def v1_repetition_penalty_(logits, prompt_mask, output_mask, repetition_penalties):
    penalties = repetition_penalties.unsqueeze(dim=1).repeat(1, logits.size(1))
    # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
    scaling = torch.where(prompt_mask | output_mask, penalties, 1.0)
    # If logits are positive, divide by penalty, otherwise multiply by penalty.
    logits *= torch.where(logits > 0, 1.0 / scaling, scaling)
    return None
