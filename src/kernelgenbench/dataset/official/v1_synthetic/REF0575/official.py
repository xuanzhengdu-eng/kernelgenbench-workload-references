# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/attention/vision.py#L363-L383
from sglang.srt.layers.attention.vision import VisionSdpaAttention

run = VisionSdpaAttention.generate_patch_attention_mask
entry_class = VisionSdpaAttention
