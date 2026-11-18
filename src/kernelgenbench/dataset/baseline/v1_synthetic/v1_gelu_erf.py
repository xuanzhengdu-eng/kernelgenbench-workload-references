# v1_gelu_erf: vLLM GELU.forward_native (native (erf)); 723-row 12.
"""vLLM GELU.forward_native reference, transcribed from the pinned body.

Source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/activation.py#L365-L366

Contract: exact erf GELU on the input dtype, out-of-place. This is vLLM's default approximate='none'; the CPU bf16 table variant is a separate item because it differs on denormal inputs.
"""
import torch
def v1_gelu_erf(x):
    return torch.nn.functional.gelu(x, approximate="none")
