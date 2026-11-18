# Source-checked synthetic-workload benchmark reference.
"""OP-802a75f20cad: vLLM sequence_parallel_chunk_impl for a contiguous 2-D input.

Pad rows to a TP multiple with zeros, select this rank's contiguous chunk, never
alias the original input and never modify it.  The process-group size/rank are
scalar inputs, exactly the two values the official implementation reads through
``get_tensor_model_parallel_world_size()`` / ``get_tensor_model_parallel_rank()``.

Source: https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/models/utils.py#L1322-L1340
"""
import torch


def v1_sequence_parallel_chunk(x, tp_size, tp_rank):
    n = x.shape[0]
    padded_n = ((n + tp_size - 1) // tp_size) * tp_size
    if padded_n != n:
        # Same zero padding as nn.functional.pad(x, (0, 0, 0, pad_len)).
        padded = torch.cat((x, x.new_zeros((padded_n - n, x.shape[1]))), dim=0)
        return padded.narrow(0, tp_rank * (padded_n // tp_size), padded_n // tp_size)
    # narrow() is a view; clone because a functional op must not return a view of x.
    return x.narrow(0, tp_rank * (n // tp_size), n // tp_size).clone()
