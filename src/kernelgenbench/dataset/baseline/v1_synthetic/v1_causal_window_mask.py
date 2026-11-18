"""Megatron causal or bottom-right aligned local attention mask.

Official sources:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/utils.py#L33-L35
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/utils.py#L38-L45

Returns a boolean (sq, skv) mask where True means "masked out".  With
``window_size=None`` the plain causal mask requires sq == skv and is the strict
upper triangle (the official ``get_default_causal_mask``).  Otherwise the
bottom-right aligned local window keeps
``skv-sq-w0 <= j-i <= skv-sq+w1`` and masks everything outside it (the official
``get_sliding_window_causal_mask``).
"""
import torch
def v1_causal_window_mask(sq,skv,window_size,device):
    i=torch.arange(sq,device=device)[:,None];j=torch.arange(skv,device=device)[None,:]
    if window_size is None:
        assert sq==skv
        return j>i
    low=skv-sq-window_size[0];high=skv-sq+window_size[1]
    return (j-i<low)|(j-i>high)
