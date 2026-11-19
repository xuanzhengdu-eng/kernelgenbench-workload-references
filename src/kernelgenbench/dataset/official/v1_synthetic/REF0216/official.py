# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/rotary_pos_embedding.py#L150-L206
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/rope_utils.py#L48-L70
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank

run = RotaryEmbedding
get_emb = RotaryEmbedding.get_emb
forward = RotaryEmbedding.forward
cp_slice = get_pos_emb_on_this_cp_rank
