# v1_squared_relu: Megatron squared_relu (native (pow)); 723-row 21.
"""Megatron squared_relu reference, transcribed from the pinned body.

Source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/activations.py#L8-L11

Contract: pow(relu(x), 2). Forward is bitwise equal to vLLM's square(relu) and SGLang's multiply form. The three share one item: measured with the pinned operators, the only divergence is the fp16 input gradient at an overflowing product (x=40000, upstream 0.1 gives 8000 for the jit-fused Megatron operator and for SGLang's multiply, but inf for vLLM's eager square), which depends on the autograd decomposition of the execution engine rather than on the operator, so the graded grid uses finite gradients and the edge case is recorded as evidence instead of being asserted.
"""
import torch
def v1_squared_relu(x):
    return torch.pow(torch.nn.functional.relu(x), 2)
