# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/cross_entropy.py#L95-L105
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/cross_entropy.py#L64-L92
import deepspeed.sequence.cross_entropy as _vce

v1_megatron_vocab_parallel_ce_backward_deepspeed = _vce._VocabParallelCrossEntropy.backward
