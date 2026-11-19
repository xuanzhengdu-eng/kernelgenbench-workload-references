# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/bias_add.py#L13-L26
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/spatial/csrc/pt_binding.cpp#L58-L79
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/spatial/csrc/opt_bias_add.cu#L50-L78
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/spatial/csrc/opt_bias_add.cu#L117-L144
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/spatial/csrc/pt_binding.cpp#L12-L32
from deepspeed.ops.transformer.inference.bias_add import nhwc_bias_add as run
