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

toks = model.to_tokens(prompt)[0]
n_toks = len(toks)
print(f'prompt = {prompt}')
print(f'n_toks = {n_toks}')

predictions, cache = model.run_with_cache(prompt)

llayer = 4
lhead = 0
letter = 'k'    # K, Q or V
letter_w = f'W_{letter.upper()}'
letter_b = f'b_{letter.upper()}'

d_model = model.cfg.d_model
center_matrix = torch.eye(d_model, device=device) - torch.ones((d_model,d_model), device=device) / d_model

resid_start = cache['blocks.0.hook_resid_pre'][0]
resid_llayer = cache[f'blocks.{llayer}.hook_resid_pre'][0]
inp_llayer = cache[f'blocks.{llayer}.attn.hook_{letter}'][0,:,lhead,:]
inp_matrix = model.blocks[llayer].attn.get_parameter(letter_w).data[lhead,:,:]
inp_bias = model.blocks[llayer].attn.get_parameter(letter_b).data[lhead,:]

ln1_scale = cache[f'blocks.{llayer}.ln1.hook_scale'][0,:] ** -1
inp_stuff = center_matrix @ inp_matrix

attn_out = []
mlp_out = []
attn_pseudo_inp = []
mlp_pseudo_inp = []
for layer in range(llayer):
    a = cache[f'blocks.{layer}.hook_attn_out'][0]
    attn_out.append(a)
    m = cache[f'blocks.{layer}.hook_mlp_out'][0]
    mlp_out.append(m)
    attn_pseudo_inp.append((a @ inp_stuff) * ln1_scale)
    mlp_pseudo_inp.append((m @ inp_stuff) * ln1_scale)

debug_sum_start = (resid_start @ inp_stuff) * ln1_scale + inp_bias
debug_sum = debug_sum_start
for layer in range(llayer):
    debug_sum = debug_sum + attn_pseudo_inp[layer] + mlp_pseudo_inp[layer]
print(debug_sum[:4,:4])
print(inp_llayer[:4,:4])
print(((resid_llayer @ inp_stuff) * ln1_scale + inp_bias)[:4,:4])

viz = np.zeros((n_toks, 2 * llayer+1))
viz[:,0] = debug_sum_start.norm(dim=1).cpu().numpy()
for layer in range(llayer):
    viz[:,2*layer+1] = attn_pseudo_inp[layer].norm(dim=1).cpu().numpy()
    viz[:,2*layer+2] = mlp_pseudo_inp[layer].norm(dim=1).cpu().numpy()

plt.imshow(viz)
plt.show()

