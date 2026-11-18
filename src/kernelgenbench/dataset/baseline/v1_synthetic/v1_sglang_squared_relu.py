# v1_sglang_squared_relu: SGLang ReLU2.forward_native (native (multiply)); 723-row 21.
"""SGLang ReLU2.forward_native reference, transcribed from the pinned body.

Source (pinned commit):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/activation.py#L286-L288

Contract: r = relu(x); r * r. Forward is bitwise equal to Megatron's pow and vLLM's square, and the fp16 gradient agrees with the jit-fused Megatron operator (8000 at x=40000, upstream 0.1), so the three references share one item. The earlier split came from comparing eager-only gradients and is withdrawn.
"""
import torch
def v1_sglang_squared_relu(x):
    relu = torch.nn.functional.relu(x)
    return relu * relu
