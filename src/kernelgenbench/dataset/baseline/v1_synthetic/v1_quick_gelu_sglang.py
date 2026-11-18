# v1_quick_gelu_sglang: SGLang QuickGELU.forward_native (native (sigmoid)); 723-row 18.
"""SGLang QuickGELU.forward_native reference, transcribed from the pinned body.

Source (pinned commit):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/activation.py#L299-L300

Contract: same body as Megatron quick_gelu and bitwise equal on every tested dtype; SGLang's forward_cuda calls forward_native, so there is no second backend.
"""
import torch
def v1_quick_gelu_sglang(x):
    return x * torch.sigmoid(1.702 * x)
