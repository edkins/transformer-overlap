import numpy as np
import transformer_lens
import torch
import matplotlib
from matplotlib import pyplot as plt

prompt = "This is so confusing and I don't know what to do."

torch.set_grad_enabled(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = transformer_lens.HookedTransformer.from_pretrained('gpt2-small', device=device)
model.set_use_attn_result(True)   # says it can burn through gpu memory
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
d_model = model.cfg.d_model
d_head = model.cfg.d_head
print(f'n_layers = {n_layers}, n_heads = {n_heads}, d_model = {d_model}, d_head = {d_head}')

tokens = model.to_tokens(prompt)[0]
n_tokens = len(tokens)
print(f'Number of tokens: {n_tokens}')
_, cache = model.run_with_cache(prompt)

fig, ax = plt.subplots(1, 1)

stuff = np.zeros((2 * n_layers + 1, d_model))
for layer in range(n_layers):
    resid = cache[f'blocks.{layer}.hook_resid_pre'][0, :, :].cpu().numpy()[1:,:]
    stuff[2*layer,:] = np.linalg.norm(resid, axis=0)
    resid = cache[f'blocks.{layer}.hook_resid_mid'][0, :, :].cpu().numpy()[1:,:]
    stuff[2*layer+1,:] = np.linalg.norm(resid, axis=0)
resid = cache[f'blocks.{n_layers-1}.hook_resid_post'][0, :, :].cpu().numpy()[1:,:]
stuff[2*n_layers,:] = np.linalg.norm(resid, axis=0)

stuff = np.sort(stuff, axis=1)

n_subbars = d_model
color = (['blue'] * n_subbars + ['green'] * n_subbars) * n_layers + ['blue'] * n_subbars

ax.bar(np.arange((2*n_layers+1) * n_subbars), stuff.reshape(-1), 1,color=color)
plt.show()
