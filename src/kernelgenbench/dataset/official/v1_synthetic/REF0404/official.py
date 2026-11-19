# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/cross_entropy.py#L107-L131
from deepspeed.sequence.cross_entropy import _GatherSequenceLoss


def run(grad_output, ctx):
    return _GatherSequenceLoss.backward(ctx, grad_output)
