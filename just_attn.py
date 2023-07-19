import numpy as np
import sys
import transformer_lens
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD

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

if len(sys.argv) > 2:
    min_layer = int(sys.argv[2])
    min_head = int(sys.argv[3])
    max_layer = min_layer + 1
    max_head = min_head + 1
else:
    min_layer = 0
    min_head = 0
    max_layer = n_layers
    max_head = n_heads

toks = model.to_tokens(prompt)[0]
n_toks = len(toks)
print(f'prompt = {prompt}')
print(f'n_toks = {n_toks}')
for i,t in enumerate(toks):
    print(f'{i}: {model.tokenizer.decode(t)}')

predictions, cache = model.run_with_cache(prompt)
fig, ax = plt.subplots(max_layer - min_layer, max_head - min_head, squeeze=False)
for layer in range(min_layer, max_layer):
    for head in range(min_head, max_head):
        a = cache[f'blocks.{layer}.attn.hook_pattern'][0][head,:,:]
        the_ax = ax[layer - min_layer, head - min_head]
        the_ax.imshow(a[1:,1:] ** 0.5, interpolation='nearest', cmap='gray')
        the_ax.set_xticks([])
        the_ax.set_yticks([])
        the_ax.set_title(f'L{layer} H{head}')
plt.show()
