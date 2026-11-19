# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py#L100-L129
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/dcp/comm.py#L292-L333
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    bcg_dcp_extend_kv_gather as run,
)
from sglang.srt.layers.dcp.comm import (
    all_gather_kv_cache_for_mla_extend as gather_entry,
)
