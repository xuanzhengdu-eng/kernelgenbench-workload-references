"""Single-rank vLLM EPLB expert-replica (physical-to-logical) mapping.

Official source (single rank specialization: ``num_groups = num_logical``,
``num_nodes = 1``, ``num_ranks = 1``):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/eplb/policy/default.py#L273-L332
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/eplb/policy/default.py#L75-L100
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/eplb/policy/default.py#L191-L271

Contract (``DefaultEplbPolicy.rebalance_experts`` with every logical expert in
its own group, one node and one rank; computes on CPU from ``weight.float()``):
- ``weight`` is the CPU/any-device load matrix ``[layers, num_logical_experts]``,
  finite and non-negative; ``num_replicas >= num_logical_experts`` is the total
  number of physical slots.
- Replication: the first ``num_logical_experts`` physical slots hold logical
  experts 0..E-1; each extra replica is assigned to the logical expert with the
  largest ``load / replica_count`` (NumPy ``argmax``, first index on ties).
- Ordering: physical slots are sorted by descending effective load
  ``load / replica_count`` using NumPy ``argsort(-effective)`` and renumbered.
- ``old_global_expert_indices`` (optional ``[layers, num_replicas]`` int) keeps
  experts that stay on the single rank in their previous slot when possible;
  incoming experts fill the remaining slots in ascending order.
- Returns an int64 CPU tensor ``[layers, num_replicas]``; no weights are moved or
  modified and the input tensors are untouched.
"""
import numpy as np
import torch


def v1_vllm_eplb_single_rank_map(weight, num_replicas, old_global_expert_indices=None):
    assert weight.ndim == 2 and weight.shape[0] > 0 and weight.shape[1] > 0
    assert torch.is_floating_point(weight) and torch.isfinite(weight).all()
    assert (weight >= 0).all() and num_replicas >= weight.shape[1]
    layers, experts = weight.shape
    old = old_global_expert_indices
    if old is not None:
        assert old.shape == (layers, num_replicas) and old.dtype in (torch.int32, torch.int64)
        assert ((old >= 0) & (old < experts)).all()
        old = old.cpu().tolist()
    loads = weight.float().cpu().numpy()
    result = []
    for layer in range(layers):
        # Step 1: replicate each logical expert up to num_replicas physical slots.
        replicas = list(range(experts))
        counts = [1] * experts
        while len(replicas) < num_replicas:
            score = loads[layer] / np.asarray(counts, dtype=np.int64)
            expert = int(np.argmax(score))
            replicas.append(expert)
            counts[expert] += 1
        # Step 2: order physical slots by descending effective load.
        effective = np.asarray(
            [float(loads[layer, expert]) / counts[expert] for expert in replicas],
            dtype=np.float64,
        )
        order = np.argsort(-effective)
        new = [replicas[index] for index in order]
        # Step 3: preserve single-rank slot positions that already match.
        if old is not None:
            used = [False] * num_replicas
            placed = [None] * num_replicas
            for slot, expert in enumerate(old[layer]):
                for index, value in enumerate(new):
                    if not used[index] and value == expert:
                        placed[slot] = value
                        used[index] = True
                        break
            remaining = iter(new[index] for index in range(num_replicas) if not used[index])
            new = [next(remaining) if value is None else value for value in placed]
        result.append(new)
    return torch.tensor(result, dtype=torch.int64, device="cpu")
