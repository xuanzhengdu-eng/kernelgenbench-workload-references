"""SGLang Gemma3 scaled word embedding.

https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/models/gemma3_causal.py#L535-L550
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/models/gemma3_causal.py#L565-L575

``Gemma3TextScaledWordEmbedding`` is an ``nn.Embedding`` subclass whose
``forward`` is ``super().forward(input_ids) * self.embed_scale``.  The scale is
a Python float fixed at construction (``hidden_size ** 0.5`` for Gemma3), the
gather itself is ``F.embedding`` and nothing is written back.  Returns
``input_ids.shape + (embedding_dim,)`` in the table dtype.
"""
import torch.nn.functional as F


def v1_sglang_embedding_lookup(table, indices, scale):
    return F.embedding(indices, table) * scale
