"""Invert a physical-to-logical expert map, preserving physical slot order.

Official source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/eplb/eplb_state.py#L1316-L1390

The input is a CPU integer tensor of shape ``(P,)`` or ``(L, P)``.  Logical ids
``>= 0`` count as active; any negative value is an inactive physical slot.  The
output is ``(E, R)`` / ``(L, E, R)`` plus a replica count ``(E,)`` / ``(L, E)``,
where ``R`` is the maximum replica count over layers.  The source derives both
with ``scatter_add_`` for the counts and a stable sort of the map for the
inverse, so physical slots of one logical expert stay in ascending physical
order and unused replicas are ``-1``.  The dtype follows the input (int32 and
int64 both work), the input is never modified, and a logical id outside
``[0, num_logical_experts)`` fails inside the scatter.
"""
import torch


def v1_expert_logical_maps(physical_to_logical_map, num_logical_experts):
    x = physical_to_logical_map
    assert x.device.type == "cpu"
    dtype = x.dtype
    per_layer = x.dim() == 1
    layers = x.unsqueeze(0) if per_layer else x
    assert len(layers.shape) == 2
    num_layers, _ = layers.shape

    valid = layers >= 0
    counts = torch.zeros((num_layers, num_logical_experts), dtype=dtype, device=x.device)
    counts.scatter_add_(1, layers.clamp(min=0), valid.to(dtype))

    max_replicas = int(counts.max().item())
    inverse = torch.full(
        (num_layers, num_logical_experts, max_replicas), -1, dtype=dtype, device=x.device
    )

    # A stable sort preserves physical-slot order inside each logical expert.
    logical_ids, physical_slots = torch.sort(layers, dim=1, stable=True)
    valid_layers, sorted_positions = (logical_ids >= 0).nonzero(as_tuple=True)
    valid_experts = logical_ids[valid_layers, sorted_positions]
    offsets = counts.cumsum(dim=1) - counts
    inactive = (~valid).sum(dim=1)
    replica_indices = (
        sorted_positions
        - inactive[valid_layers]
        - offsets[valid_layers, valid_experts]
    )
    inverse[valid_layers, valid_experts, replica_indices] = physical_slots[
        valid_layers, sorted_positions
    ].to(dtype)
    if per_layer:
        return inverse.squeeze(0), counts.squeeze(0)
    return inverse, counts
