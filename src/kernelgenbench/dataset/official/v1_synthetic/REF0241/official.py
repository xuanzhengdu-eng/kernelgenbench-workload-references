# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/chunked_cross_entropy.py#L159-L179
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/chunked_cross_entropy.py#L198-L239
from deepspeed.runtime.chunked_cross_entropy import chunked_cross_entropy


def run(logits, target, ignore_index=-100, backend="auto"):
    return chunked_cross_entropy(
        logits,
        target,
        ignore_index=ignore_index,
        reduction="none",
        backend=backend,
    )
