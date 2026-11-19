# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py#L106-L158
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py#L218-L272
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)

__all__ = ["YarnRotaryEmbedding", "build", "run"]


def build(args, device):

    return YarnRotaryEmbedding(
        kv_channels=int(args["dim"]),
        rotary_percent=1.0,
        rotary_interleaved=False,
        seq_len_interpolation_factor=None,
        rotary_base=float(args["base"]),
        use_cpu_initialization=False,
        scaling_factor=float(args["scaling"]),
        original_max_position_embeddings=int(args["original_max"]),
        beta_fast=float(args["beta_fast"]),
        beta_slow=float(args["beta_slow"]),
        mscale=float(args["mscale"]),
        mscale_all_dim=float(args["mscale_all_dim"]),
        correction_range_round_to_int=bool(args["round_range"]),
        cp_group=None,
    ).to(device)


def run(module, seq_len, offset=0):

    return module.get_emb(int(seq_len), int(offset))
