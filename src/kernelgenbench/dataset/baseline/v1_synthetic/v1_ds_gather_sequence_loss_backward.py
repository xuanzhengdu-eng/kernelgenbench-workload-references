"""DeepSpeed ``_GatherSequenceLoss.backward`` with ``sum_gradients=False``.

The forward all-gathers the per-rank loss rows; the backward keeps this rank's
dim-0 window of the incoming gradient (a narrow view, even for one rank) and
returns three slots: gradient, ``sp_group``, ``sum_gradients``.
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/cross_entropy.py#L107-L131
"""
import torch

def v1_ds_gather_sequence_loss_backward(grad_output,group):
    world=group.size()
    assert grad_output.shape[0]%world==0
    chunk=grad_output.shape[0]//world
    return grad_output.narrow(0,group.rank()*chunk,chunk),None,None
