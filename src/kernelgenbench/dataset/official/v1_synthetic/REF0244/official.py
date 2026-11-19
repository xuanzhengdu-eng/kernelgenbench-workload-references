# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/vocab_parallel_ops.py#L306-L335
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/vocab_parallel_ops.py#L437-L450
from deepspeed.sequence.vocab_parallel_ops import vocab_parallel_kl_div


def run(student, teacher, reverse=False, tp_group=None):
    return vocab_parallel_kl_div(student, teacher, reverse=reverse, tp_group=tp_group)
