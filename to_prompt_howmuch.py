import numpy as np
import transformer_lens
import torch
from matplotlib import pyplot as plt

prompt = 'The quick brown fox jumps over the lazy dog.'

torch.set_grad_enabled(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = transformer_lens.HookedTransformer.from_pretrained('gpt2-small', device=device)
print(f'n_layers = {model.cfg.n_layers}, n_heads = {model.cfg.n_heads}, d_model = {model.cfg.d_model}')

toks = model.tokenizer.encode(prompt)
print(f'prompt = {prompt}')
print(f'n_toks = {len(toks)}')

predictions, cache = model.run_with_cache([prompt])
print(f'predictions.shape = {predictions.shape}')
#print(f'cache.keys() = {cache.keys()}')

llayer = 4
lhead = 0
letter = 'k'    # K, Q or V
letter_w = f'W_{letter.upper()}'
letter_b = f'b_{letter.upper()}'

resid_start = cache['blocks.0.hook_resid_pre'][0]
resid_llayer = cache[f'blocks.{llayer}.hook_resid_pre'][0]
inp_llayer = cache[f'blocks.{llayer}.attn.hook_{letter}'][0,:,lhead,:]
inp_matrix = model.blocks[llayer].attn.get_parameter(letter_w).data[lhead,:,:]
inp_bias = model.blocks[llayer].attn.get_parameter(letter_b).data[lhead,:]
#inp_ln1 = model.blocks[llayer].ln1
#inp_normalized = cache[f'blocks.{llayer}.ln1.hook_normalized'][0,:,:]
#print(inp_ln1)

ln1_scale = cache[f'blocks.{llayer}.ln1.hook_scale'][0,:]

attn_out = []
mlp_out = []
attn_pseudo_inp = []
mlp_pseudo_inp = []
for layer in range(llayer):
    a = cache[f'blocks.{layer}.hook_attn_out'][0]
    attn_out.append(a)
    m = cache[f'blocks.{layer}.hook_mlp_out'][0]
    mlp_out.append(m)
    attn_pseudo_inp.append((a @ inp_matrix))
    mlp_pseudo_inp.append((m @ inp_matrix))

def ln1(x):
    x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, length]
    return x / ln1_scale

debug_sum = resid_start @ inp_matrix
for layer in range(llayer):
    debug_sum = debug_sum + attn_pseudo_inp[layer] + mlp_pseudo_inp[layer]
print(debug_sum.shape, inp_llayer.shape)
print((ln1(debug_sum) + inp_bias)[:4,:4])
print(inp_llayer[:4,:4])
print((ln1(resid_llayer) @ inp_matrix + inp_bias)[:4,:4])