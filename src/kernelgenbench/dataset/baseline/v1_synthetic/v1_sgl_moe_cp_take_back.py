"""SGLang MoE-CP take-back: undo the padded gather and optionally scatter to DP.

Official sources (sgl-project/sglang, pinned commit 62384de1):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/layer_boundary/ops.py#L334-L342
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/layer_boundary/ops.py#L470-L492

Explicit-argument sub-contract of ``moe_cp_take_back_output``:

* ``moe_cp_take_back``: the gathered MoE output is padded to the largest CP
  shard, so this rank's rows start at ``moe_cp_rank * max(rows)`` and are
  ``rows[moe_cp_rank]`` long; the slice is made contiguous.
* the optional DP scatter (see ``dp_attention.dp_scatter``): a preallocated
  ``dp_output`` is zeroed, this DP slot's ``[dp_start, dp_start+dp_count)`` rows
  of the CP-local buffer are copied into it, and it becomes the result.
* ``residual`` is passed through unchanged.
"""
import torch


def v1_sgl_moe_cp_take_back(hidden_states, residual, rows, moe_cp_rank, dp_output=None, dp_start=0, dp_count=0):
    if rows is not None:
        start = moe_cp_rank * max(rows)
        count = rows[moe_cp_rank]
        hidden_states = hidden_states.narrow(0, start, count).contiguous()
    if dp_output is not None:
        assert hidden_states.is_contiguous() and dp_output.is_contiguous()
        dp_output.fill_(0)
        dp_output[:dp_count].copy_(hidden_states[dp_start:dp_start + dp_count])
        hidden_states = dp_output
    return hidden_states, residual
