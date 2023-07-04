import numpy as np
import transformer_lens
import torch
import matplotlib
from matplotlib import pyplot as plt
from sklearn.decomposition import DictionaryLearning

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

fig, ax = plt.subplots(n_heads, 1)
layer = 2
X = np.zeros((n_layers * n_heads * n_tokens, d_model))
for head in range(n_heads):
    X[head * n_tokens:(head + 1) * n_tokens, :] = cache[f'blocks.{layer}.attn.hook_result'][0, :, head, :].cpu()

n_components = 100
learn = DictionaryLearning(n_components=n_components, verbose=True)
Xt = learn.fit_transform(X)

for head in range(n_heads):
    viz = np.zeros((n_tokens, n_components))
    viz[:, :] = Xt[head * n_tokens:(head + 1) * n_tokens, :]

    # hide axes and ticks
    ax[head].axis('off')
    #ax[head].set_xticks([])
    #ax[head].set_yticks([])

    ax[head].imshow(viz, aspect='auto', cmap='coolwarm', norm=matplotlib.colors.CenteredNorm())
plt.show()
