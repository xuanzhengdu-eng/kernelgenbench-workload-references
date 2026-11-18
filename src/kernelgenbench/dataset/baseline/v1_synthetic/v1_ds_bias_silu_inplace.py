"""DeepSpeed SILU inference v2: [tokens, channels], optional bias, in-place update, None return."""
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation.py#L21-L62
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation_cuda.cu#L36-L87
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation.cpp#L38-L68
import torch


def v1_ds_bias_silu_inplace(activation, bias=None):
    # The native kernel reads aligned 16-byte channel vectors via raw pointers.
    if activation.ndim != 2 or activation.numel() == 0:
        raise ValueError("expected a nonempty rank-2 activation")
    if activation.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("unsupported activation dtype")
    channels = activation.shape[-1]
    if channels * activation.element_size() % 16:
        raise ValueError("channel width must be 16-byte aligned")
    if activation.requires_grad or not activation.is_contiguous():
        raise ValueError("expected contiguous inference activation")
    if activation.is_contiguous() and activation.data_ptr() % 16:
        raise ValueError("unaligned activation address")
    if bias is not None:
        if (bias.shape != (channels,) or bias.dtype != activation.dtype
                or bias.device != activation.device or not bias.is_contiguous()
                or bias.requires_grad or bias.data_ptr() % 16):
            raise ValueError("expected aligned channel bias of matching dtype and device")
    out = activation
    x = out.float() + (0.0 if bias is None else bias.float())
    result = x / (1.0 + torch.exp(-x))
    out.copy_(result.to(out.dtype))
