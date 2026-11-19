# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/rotary_embedding/rope_variant.py#L472-L505
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)


forward_native = DeepseekScalingRotaryEmbedding.forward_native

__all__ = ["DeepseekScalingRotaryEmbedding", "forward_native"]
