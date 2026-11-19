# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/qkv_proj.cpp#L623-L690
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/qkv_proj.cpp#L416-L620
import os

import torch

_EXTENSION = os.environ.get(
    "KGB_SGL_CPU_EXT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "common_ops.so")
)
torch.ops.load_library(_EXTENSION)


def run(
    hidden_states,
    qkv_a_proj_weight,
    q_b_proj_weight,
    w_kc,
    q_a_layernorm_weight,
    kv_a_layernorm_weight,
    positions,
    cos_sin_cache,
    eps,
    use_int8_w8a8,
    use_fp8_w8a16,
    qkv_a_proj_scale,
    q_b_proj_scale,
    w_scale,
    is_vnni,
    block_size,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_head_dim,
):
    return torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight(
        hidden_states,
        qkv_a_proj_weight,
        q_b_proj_weight,
        w_kc,
        q_a_layernorm_weight,
        kv_a_layernorm_weight,
        positions,
        cos_sin_cache,
        eps,
        use_int8_w8a8,
        use_fp8_w8a16,
        qkv_a_proj_scale,
        q_b_proj_scale,
        w_scale,
        is_vnni,
        block_size,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
    )
