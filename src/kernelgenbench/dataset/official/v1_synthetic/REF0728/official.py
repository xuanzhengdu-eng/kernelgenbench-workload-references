# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/logits_processor/builtin.py#L159-L162
from vllm.v1.sample.logits_processor.builtin import LogitBiasLogitsProcessor

run = LogitBiasLogitsProcessor.apply
