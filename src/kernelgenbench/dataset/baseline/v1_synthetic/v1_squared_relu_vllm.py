# v1_squared_relu_vllm: vLLM ReLUSquaredActivation.forward_native (native (square)); 723-row 21.
"""vLLM ReLUSquaredActivation.forward_native reference, transcribed from the pinned body.

Source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/activation.py#L666-L668

Contract: torch.square(relu(x)). The CUDA relu_squared kernel matches this bitwise in fp32, fp16 and bf16, so native and CUDA are one backend pair. Same item as the Megatron operator; see that entry for the fp16 overflow-gradient evidence.
"""
import torch
def v1_squared_relu_vllm(x):
    return torch.square(torch.nn.functional.relu(x))
