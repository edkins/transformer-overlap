import numpy as np
import sys
import transformer_lens
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

prompt = sys.argv[1]

torch.set_grad_enabled(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = transformer_lens.HookedTransformer.from_pretrained('gpt2-small', device=device)
n_heads = model.cfg.n_heads
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
d_head = model.cfg.d_head
print(f'n_layers = {n_layers}, n_heads = {n_heads}, d_model = {d_model}')

toks = model.to_tokens(prompt)[0]
n_toks = len(toks)
print(f'prompt = {prompt}')
print(f'n_toks = {n_toks}')
for i,t in enumerate(toks):
    print(f'{i}: {model.tokenizer.decode(t)}')

predictions, cache = model.run_with_cache(prompt)

attn = np.zeros((n_layers, n_heads, n_toks - 1, n_toks - 1))
for layer in range(n_layers):
    attn[layer,:,:,:] = cache[f'blocks.{layer}.attn.hook_pattern'][0][:,1:,1:] ** 0.5

X = attn.reshape((n_layers * n_heads, (n_toks - 1) ** 2))
tsne = TSNE(n_components=2, perplexity=10, verbose=2)
attn_xy = tsne.fit_transform(X).reshape((n_layers, n_heads, 2))

fig = plt.figure(figsize=(16,16))
for layer in range(n_layers):
    for head in range(n_heads):
        a = attn[layer,head,:,:]
        x,y = attn_xy[layer,head,:]
        fig.figimage(a, xo=800 + 30 * x, yo=800 + 30 * y, cmap='gray')
plt.show()
