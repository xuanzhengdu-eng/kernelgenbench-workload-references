# NVIDIA Megatron, vLLM and SGLang pinned sources cover the unquantized contiguous vocab-shard sub-contract: mask non-owned IDs, embed local rows, zero masked contributions, then all-reduce full embeddings. Local omits added/padded vocab layouts, reduce-scatter, quantized and fused embedding paths; the links apply only to the plain TP all-reduce mode. No DeepInfer source was verified.
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/layers.py#L276-L303
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/vocab_parallel_embedding.py#L547-L569
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/vocab_parallel_embedding.py#L494-L529
"""Contiguous evenly sharded vocabulary lookup + TP all-reduce.
Local weight [V/world,D], identical valid ids on all ranks, optional group.
The rank-local contiguous shard starts at rank*local_rows. Both tensor-parallel
peers receive the full embedding result. Allreduce is skipped for TP=1.
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist

def v1_vocab_parallel_embedding(ids,weight,group=None):
 """ids [..], local weight [V/world, D] -> full embedding [.., D] on every rank."""
 world=1 if group is None else dist.get_world_size(group);rank=0 if group is None else dist.get_rank(group)
 if weight.ndim!=2 or not weight.shape[0] or ids.dtype not in (torch.int32,torch.int64):raise ValueError('invalid input')
 if world==1:
  # Upstream skips masking when the tensor-parallel group has one rank, so an
  # out-of-range id fails in F.embedding instead of yielding zeros.
  return F.embedding(ids.long(),weight)
 first=rank*weight.shape[0];mask=(ids<first)|(ids>=first+weight.shape[0]);idx=(ids.long()-first).masked_fill(mask,0)
 out=F.embedding(idx,weight).masked_fill(mask.unsqueeze(-1),0)
 dist.all_reduce(out,group=group)
 return out
