# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/spec_utils.py#L1104-L1163
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/spec_utils.py#L1166-L1426
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#L1606-L1711
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py#L781-L848
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py#L880-L911
from sglang.srt.speculative.spec_utils import _verify_commit_step_indices
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    scatter_mamba_states_after_mtp_verify,
)

step_indices = _verify_commit_step_indices
scatter = scatter_mamba_states_after_mtp_verify
