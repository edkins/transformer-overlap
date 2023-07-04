import numpy as np
import transformer_lens
import torch
import matplotlib
from matplotlib import pyplot as plt
from sklearn.decomposition import DictionaryLearning, PCA

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

X = np.zeros((n_heads, d_head, d_model))
X[:, :, :] = model.blocks[4].attn.W_O.data.cpu()

X = X.reshape(n_heads * d_head, d_model)

n_components = 100
learn = DictionaryLearning(n_components=n_components, verbose=True, max_iter=100)
learn.fit(X)

fig, ax = plt.subplots(n_tokens, 1)

stuff = np.zeros((n_layers, n_tokens, n_components))
for layer in range(n_layers):
    stuff[layer-3,:,:] = learn.transform(cache[f'blocks.{layer}.hook_resid_pre'][0, :, :].cpu().numpy())

for tok in range(n_tokens):
    viz = np.zeros((n_layers, n_components))
    viz[:, :] = stuff[:, tok, :]

    ax[tok].axis('off')
    ax[tok].imshow(viz, aspect='auto', cmap='coolwarm', norm=matplotlib.colors.CenteredNorm())
plt.show()
