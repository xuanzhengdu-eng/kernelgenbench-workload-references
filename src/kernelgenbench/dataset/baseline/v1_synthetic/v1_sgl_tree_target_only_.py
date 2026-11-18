"""SGLang tree target-only verification with in-place caller buffers.

Official sources (fixed commit 62384de1ae5e3bbd3159afd1639525b0bf39fbf9):
- JIT kernel ``TreeSpeculativeSamplingTargetOnly``:
  https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/jit/include/sgl_kernel/speculative/sampling.cuh#L30-L178
- Python wrapper ``tree_speculative_sampling_target_only``:
  https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/speculative/sampling.py#L33-L65

Contract:

- ``candidates`` is int64 ``[batch, nodes]``; ``retrive_index`` maps each node to
  its global slot, and ``retrive_next_token`` / ``retrive_next_sibling`` describe
  the tree (first child, next sibling, ``-1`` for none).
- Starting from the root, each verify level scans children and then siblings,
  accumulating ``p = target_probs[b, node, token]``.  A candidate is accepted when
  ``p > 0`` and ``coins[b, node] < acc / threshold_acc`` or
  ``p >= threshold_single``; a rejected sibling records its probability into
  ``draft_probs[b, distribution_node, token]``.
- On rejection the bonus token is drawn from ``relu(target - draft)`` at the
  visited node; when the whole draft is accepted, from the pure target row.  The
  draw uses the final coin and a float32 cumulative sum, falling back to the last
  positive entry and then ``vocab - 1``.
- ``predicts``, ``accept_index``, ``accept_token_num`` and ``draft_probs`` are
  caller buffers written in place; the function returns ``None``.
"""
import torch


def v1_sgl_tree_target_only_(predicts,accept_index,accept_token_num,candidates,retrive_index,retrive_next_token,retrive_next_sibling,uniform_samples,uniform_samples_for_final_sampling,target_probs,draft_probs,threshold_single,threshold_acc):
    """Target-only tree verify, explicit coins and in-place output/draft buffers."""
    batch,n=candidates.shape; spec=accept_index.shape[1]; vocab=target_probs.shape[-1]
    assert batch>0 and n>0 and spec>0 and 0<threshold_acc<=1 and 0<=threshold_single<=1
    for b in range(batch):
        cur=0;last=int(retrive_index[b,0]);accept_index[b,0]=last;accepted=0;acc=0.;coin=uniform_samples[b,0];distribution_node=0
        for step in range(1,spec):
            cur=int(retrive_next_token[b,cur])
            while cur!=-1:
                token=int(candidates[b,cur]);p=target_probs[b,distribution_node,token];acc=acc+float(p)
                if p>0 and (coin<acc/threshold_acc or p>=threshold_single):
                    acc=0.;distribution_node=cur;coin=uniform_samples[b,cur];predicts[last]=token;accepted+=1;last=int(retrive_index[b,cur]);accept_index[b,accepted]=last;break
                draft_probs[b,distribution_node,token]=p;cur=int(retrive_next_sibling[b,cur])
            if cur==-1:break
        accept_token_num[b]=accepted
        weights=target_probs[b,distribution_node] if accepted==spec-1 else (target_probs[b,distribution_node]-draft_probs[b,distribution_node]).clamp(min=0)
        cum=weights.cumsum(0);idx=(cum>uniform_samples_for_final_sampling[b]*cum[-1]).nonzero().flatten()
        if idx.numel():token=idx[0]
        else:
            valid=(weights>0).nonzero().flatten();token=valid[-1] if valid.numel() else vocab-1
        predicts[last]=token
