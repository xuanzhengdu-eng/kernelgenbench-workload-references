# Official entry for REF0695 v1_sgl_cp_zigzag_gather.
#
# Framework: SGLang, pinned commit 62384de1ae5e3bbd3159afd1639525b0bf39fbf9.
# Backend: CP all-gather through get_parallel().attn_cp_group (Gloo process
# group in this replay), zigzag strategy.
#
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/cp/zigzag.py#L112-L115
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/cp/zigzag.py#L366-L375
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/cp/zigzag.py#L510-L547
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy

# Unbound official method: call as run(strategy, x, forward_batch).
run = ZigzagCPStrategy.gather_hidden_states

ENTRY = "sglang.srt.layers.cp.zigzag.ZigzagCPStrategy.gather_hidden_states"
STRATEGY_CLASS = ZigzagCPStrategy
