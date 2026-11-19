# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/mlp.py#L246-L354
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/shared_experts.py#L121-L129
from megatron.core.transformer.mlp import MLP, MLPSubmodules


from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

__all__ = ["MLP", "MLPSubmodules", "SharedExpertMLP", "build", "run"]


def _tp_group():

    import os
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(
            "gloo",
            init_method="tcp://127.0.0.1:%d" % (29500 + os.getpid() % 400),
            rank=0,
            world_size=1,
        )
    return dist.group.WORLD


def _spec():

    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    return MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear)


def build(x, w1, w2, b1=None, gate_weight=None, linear_offset=0.0, clamp_value=None):





    import torch
    import torch.nn.functional as F

    from megatron.core.transformer.transformer_config import TransformerConfig

    ffn = w1.shape[0] // 2
    hidden = w2.shape[0]
    on_cuda = x.is_cuda
    if on_cuda:
        torch.cuda.set_device(x.device)

    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=1,
        ffn_hidden_size=ffn,
        moe_shared_expert_intermediate_size=ffn,
        gated_linear_unit=True,
        add_bias_linear=False,
        activation_func=F.silu,
        glu_linear_offset=linear_offset,
        activation_func_clamp_value=clamp_value,
        bias_activation_fusion=False,
        params_dtype=x.dtype,
        bf16=(x.dtype == torch.bfloat16),
        fp16=(x.dtype == torch.float16),
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        use_cpu_initialization=not on_cuda,
        perform_initialization=False,
    )
    group = _tp_group()
    gate = gate_weight is not None

    if gate:
        from megatron.core.process_groups_config import ProcessGroupCollection

        pgs = ProcessGroupCollection()
        pgs.tp = group
        module = SharedExpertMLP(cfg, _spec(), gate=True, pg_collection=pgs)
    else:
        module = MLP(cfg, _spec(), input_size=hidden, ffn_hidden_size=ffn, tp_group=group)

    module = module.to(device=x.device, dtype=x.dtype)
    with torch.no_grad():
        module.linear_fc1.weight.copy_(w1)
        module.linear_fc2.weight.copy_(w2)
        if gate:
            module.gate_weight.copy_(gate_weight)
        if b1 is not None and getattr(module.linear_fc1, "bias", None) is not None:
            module.linear_fc1.bias.copy_(b1)
    module.eval()

    def forward(return_bias=False):
        with torch.no_grad():
            out, bias = module.forward(x)
        return (out, bias) if return_bias else out

    return module, forward


def run(x, w1, w2, b1=None, gate_weight=None, linear_offset=0.0, clamp_value=None):

    _module, forward = build(
        x=x, w1=w1, w2=w2, b1=b1, gate_weight=gate_weight,
        linear_offset=linear_offset, clamp_value=clamp_value,
    )
    return forward()
