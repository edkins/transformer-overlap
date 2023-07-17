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
model.set_use_attn_result(True)   # says it can burn through gpu memory
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
fig, ax = plt.subplots(n_layers, n_heads)
for layer in range(n_layers):
    for head in range(n_heads):
        k = cache[f'blocks.{layer}.attn.hook_k'][0,:,head,:]
        q = cache[f'blocks.{layer}.attn.hook_q'][0,:,head,:]
        a = cache[f'blocks.{layer}.attn.hook_pattern'][0][head,:,:]
        kq = torch.nn.functional.normalize(torch.concatenate((
            k.reshape(1, n_toks, d_head).expand(n_toks, n_toks, d_head),
            q.reshape(n_toks, 1, d_head).expand(n_toks, n_toks, d_head),
        ), dim=2), dim=2)
        n_toks1 = n_toks - 1
        kqa = (kq * a.reshape(n_toks, n_toks, 1))[1:,1:,:]  # remove first token
        reducer = TruncatedSVD(n_components=3)
        kqat = reducer.fit_transform(kqa.reshape(n_toks1 * n_toks1, 2 * d_head))
        colors = np.clip(0.25 + 2 * kqat.reshape(n_toks1, n_toks1, 3), 0, 1)
        ax[layer, head].imshow(colors)
        ax[layer, head].set_xticks([])
        ax[layer, head].set_yticks([])
        ax[layer, head].set_title(f'L{layer} H{head}')
plt.show()
