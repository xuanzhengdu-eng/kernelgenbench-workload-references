"""vLLM unquantized MLA KV up-projection dtype conversion and views.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/attention/mla_attention.py#L409-L425
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/quant_utils.py#L528-L550
"""
def v1_vllm_split_kv_projection(weight,out_dtype,kv_lora_rank,num_heads,qk_nope_head_dim,v_head_dim):
    x=weight.to(out_dtype).T.view(kv_lora_rank,num_heads,qk_nope_head_dim+v_head_dim)
    return x.split([qk_nope_head_dim,v_head_dim],dim=-1)
