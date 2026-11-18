# v1_sglang_geglu_and_mul: SGLang GeluAndMul.forward_native (native (tanh)); 723-row 16.
"""SGLang GeluAndMul.forward_native reference, transcribed from the pinned body.

Source (pinned commit):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/activation.py#L240-L242

Contract: same width//2 slicing as vLLM but the constructor default is approximate='tanh', so the default behavior differs from vLLM's default despite identical slicing.
"""
import torch
def v1_sglang_geglu_and_mul(x):
    width = x.shape[-1] // 2
    return torch.nn.functional.gelu(x[..., :width], approximate="tanh") * x[..., width:]
