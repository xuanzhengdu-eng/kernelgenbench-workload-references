# v1_geglu_back: Megatron geglu_back (native (tanh)); 723-row 15.
"""Megatron geglu_back reference, transcribed from the pinned body.

Source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_geglu.py#L49-L66

Contract: chunks the last axis in two and concatenates the gradient halves; two inputs, no bias. Odd widths split unevenly via torch.chunk. Upstream runs under jit_fuser (=torch.compile), which fuses the pointwise chain and can differ from eager execution by an element ULP or two; the graded grid keeps the inputs small enough that both execution modes stay inside the project tolerance.
"""
import torch
def v1_geglu_back(g, y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    tanh_out = torch.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1))
    ff = 0.5 * y_1 * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * y_1 * y_1)) \
        + 0.5 * (1 + tanh_out)
    return torch.cat(((g * y_2) * ff, g * (y_1 * 0.5 * (1.0 + tanh_out))), -1)
