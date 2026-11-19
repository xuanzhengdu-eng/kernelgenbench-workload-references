# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/language_model_embedding.py#L98-L154
"""Official Megatron-LM entry for REF0606, pinned commit 266f1c9.

``LanguageModelEmbedding.forward`` is the captured framework entry.  The class
itself builds its three tables in ``__init__`` (``word_embeddings`` is a
``tensor_parallel.VocabParallelEmbedding``, the position and token-type tables
are plain fp32 ``torch.nn.Embedding`` modules), so a caller must instantiate the
official class with a real ``TransformerConfig`` before driving ``forward``.
``build_embedding`` and ``run`` below only assemble that module and call it; the
tensor mapping, layout restoration and device placement live in
replay_check.py.
"""
import torch

from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.transformer.transformer_config import TransformerConfig

OFFICIAL_ENTRY = (
    "megatron.core.models.common.embeddings.language_model_embedding."
    "LanguageModelEmbedding.forward"
)


def build_embedding(vocab_size, hidden_size, max_sequence_length, params_dtype,
                    position_embedding_type="rope", num_tokentypes=0,
                    hidden_dropout=0.0, use_mup=False, mup_embedding_mult=1.0,
                    fp32_residual_connection=False, sequence_parallel=False,
                    scatter_to_sequence_parallel=False, tp_group=None):
    """Construct the pinned official module around caller-supplied tables.

    ``perform_initialization=False`` and ``use_cpu_initialization=True`` keep the
    official random init out of the way; the replay script copies the captured
    checkpoint rows in afterwards, exactly as the collector did.
    """
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=max(1, hidden_size // 128),
        params_dtype=params_dtype,
        bf16=params_dtype == torch.bfloat16,
        fp16=params_dtype == torch.float16,
        use_cpu_initialization=True,
        perform_initialization=False,
        sequence_parallel=sequence_parallel,
        hidden_dropout=hidden_dropout,
        deterministic_mode=False,
        fp32_residual_connection=fp32_residual_connection,
        use_mup=use_mup,
        mup_embedding_mult=mup_embedding_mult,
    )
    module = LanguageModelEmbedding(
        config=config,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        position_embedding_type=position_embedding_type,
        num_tokentypes=num_tokentypes,
        scatter_to_sequence_parallel=scatter_to_sequence_parallel,
        tp_group=tp_group,
    )
    return module


def run(module, input_ids, position_ids=None, tokentype_ids=None):
    """Call the official ``LanguageModelEmbedding.forward`` unchanged."""
    return module(input_ids=input_ids, position_ids=position_ids,
                  tokentype_ids=tokentype_ids)
