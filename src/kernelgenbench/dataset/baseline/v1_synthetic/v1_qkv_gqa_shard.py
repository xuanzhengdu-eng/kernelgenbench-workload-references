"""DeepSpeed fused QKV parameter sharding, GQA KV replication and MHA default.

Source (pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/model_implementations/sharding/qkv.py#L14-L128

Contract: shards a fused QKV weight ``[(n_q + 2 n_kv) * head_size, ...]`` or
bias ``[(n_q + 2 n_kv) * head_size]`` for one TP shard.  The Q rows are always
outer-dimension sharded.  With ``n_kv_heads >= num_shards`` the KV rows are
outer-dimension sharded with two concatenated matrices, so each shard keeps a
K and a V head; with ``n_kv_heads < num_shards`` each KV head is replicated to
``num_shards / n_kv_heads`` consecutive shards while that group's Q heads are
sharded inside the group.  When the head counts are equal (or omitted, which
derives them from the fused width) the MHA branch shards Q, K and V as three
concatenated matrices.  The parameter layout is Q heads, then K heads, then V
heads, all contiguous; the result is a new tensor and the input is untouched.

The module is the transcription of the pinned body; the shared
``v1_shard_parameter`` helper is the pinned ``shard_param``.
"""
import torch

from .v1_shard_parameter import v1_shard_parameter


def v1_qkv_gqa_shard(param, shard_rank, num_shards, head_size,
                     n_heads_q=None, n_heads_kv=None):
    if n_heads_kv is not None and n_heads_q is None:
        raise ValueError("n_heads_kv should not be passed without n_heads_q")
    if param is None:
        raise ValueError("param should not be None")
    if n_heads_q is None:
        if param.shape[0] // 3 % head_size != 0:
            raise ValueError("MHA param shape is not correct")
        n_heads_q = param.shape[0] // head_size // 3
        mha_sharding = True
    elif n_heads_kv is None:
        mha_sharding = True
    else:
        mha_sharding = n_heads_q == n_heads_kv

    if n_heads_q < num_shards:
        raise ValueError(
            "There must be at least as many query heads as there are shards.")

    if mha_sharding:
        return v1_shard_parameter(param, "outer", shard_rank, num_shards,
                                  3, head_size, 1)

    if n_heads_q % n_heads_kv != 0:
        raise ValueError("Must be an even ratio between query and key/value heads.")
    if param.shape[0] != head_size * (n_heads_q + 2 * n_heads_kv):
        raise ValueError("GQA param shape is not correct")
    if n_heads_kv >= num_shards and n_heads_kv % num_shards != 0:
        raise ValueError(
            "Currently do not support uneven partitioning of KV heads for GQA.")
    if n_heads_kv < num_shards and num_shards % n_heads_kv != 0:
        raise ValueError(
            "Currently do not support distributing KV heads across different "
            "numbers of shards.")
    even_kv_sharding = n_heads_kv >= num_shards

    q_param = param[:head_size * n_heads_q]
    kv_param = param[head_size * n_heads_q:]

    if even_kv_sharding:
        # n_q = C * n_kv, so n_kv % num_shards == 0 implies n_q % num_shards == 0.
        q_param = v1_shard_parameter(q_param, "outer", shard_rank, num_shards,
                                     1, head_size, 1)
        kv_param = v1_shard_parameter(kv_param, "outer", shard_rank, num_shards,
                                      2, head_size, 1)
        return torch.cat([q_param, kv_param], dim=0)

    q_sharding_degree = num_shards // n_heads_kv
    kv_head = shard_rank // q_sharding_degree
    k_param = kv_param[kv_head * head_size:(kv_head + 1) * head_size]
    v_param = kv_param[(n_heads_kv + kv_head) * head_size:
                       (n_heads_kv + kv_head + 1) * head_size]
    q_sharding_rank = shard_rank % q_sharding_degree
    q_factor = n_heads_q // n_heads_kv
    q_chunk = q_param[q_factor * kv_head * head_size:
                      q_factor * (kv_head + 1) * head_size]
    q_param = v1_shard_parameter(q_chunk, "outer", q_sharding_rank,
                                 q_sharding_degree, 1, head_size, 1)
    return torch.cat([q_param, k_param, v_param], dim=0)
