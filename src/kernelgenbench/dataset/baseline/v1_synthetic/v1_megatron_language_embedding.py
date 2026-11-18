"""Megatron-LM language-model input embedding (single rank, optional dropout).

Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/language_model_embedding.py#L99-L154

``LanguageModelEmbedding.forward`` on one tensor-parallel rank, unquantized and
without sequence-parallel scatter:

1. word embedding lookup,
2. optional learned-absolute position embedding added in [b, s, h],
3. transpose to [s, b, h],
4. optional token-type embedding added after the transpose
   (``permute(1, 0, 2)`` of the [b, s, h] lookup),
5. MuP scale when ``use_mup`` and the multiplier differs from 1.0,
6. fp32 cast when ``fp32_residual_connection`` is set,
7. ``nn.Dropout(hidden_dropout)`` -- identity in eval mode, inverted scaling when
   training.

Precision contract worth calling out: the module builds the position and
token-type tables as plain ``torch.nn.Embedding`` objects *without* a dtype, so
they are always fp32 (the word embedding follows ``config.params_dtype``).
Adding them therefore promotes the whole embedding to fp32 for fp16/bf16
models -- measurable behaviour, not a rounding detail.  The reference keeps that
by casting those two tables up before the lookup.

The reference takes the weights directly so a test can drive it without a
checkpoint; the module-level entry is verified separately against the pinned
class.
"""
import torch
import torch.nn.functional as F


def v1_megatron_language_embedding(input_ids, position_ids, tokentype_ids, word_weight,
                                   position_weight, tokentype_weight, mup_scale,
                                   fp32_residual, dropout_prob=0.0, training=False):
    embeddings = F.embedding(input_ids, word_weight)
    if position_weight is not None:
        # Upstream keeps this table in fp32 regardless of params_dtype.
        embeddings = embeddings + F.embedding(position_ids, position_weight.float())
    embeddings = embeddings.transpose(0, 1).contiguous()
    if tokentype_ids is not None:
        # Same fp32 table as upstream's nn.Embedding(num_tokentypes, hidden).
        embeddings = embeddings + F.embedding(tokentype_ids,
                                              tokentype_weight.float()).permute(1, 0, 2)
    if mup_scale != 1.0:
        embeddings = embeddings * mup_scale
    if fp32_residual:
        embeddings = embeddings.float()
    return F.dropout(embeddings, p=dropout_prob, training=training)
