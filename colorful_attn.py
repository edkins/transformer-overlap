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

alg = 2

predictions, cache = model.run_with_cache(prompt)
fig, ax = plt.subplots(max_layer - min_layer, max_head - min_head, squeeze=False)
for layer in range(min_layer, max_layer):
    for head in range(min_head, max_head):
        k = cache[f'blocks.{layer}.attn.hook_k'][0,:,head,:]
        q = cache[f'blocks.{layer}.attn.hook_q'][0,:,head,:]
        a = cache[f'blocks.{layer}.attn.hook_pattern'][0][head,:,:]

        if alg == 1:
            kq = torch.concatenate((
                k.reshape(1, n_toks, d_head).expand(n_toks, n_toks, d_head),
                q.reshape(n_toks, 1, d_head).expand(n_toks, n_toks, d_head),
            ), dim=2)
            dkq = 2 * d_head
        elif alg == 2:
            kq = k.reshape(1, n_toks, d_head) * q.reshape(n_toks, 1, d_head)
            dkq = d_head
        else:
            raise ValueError(f'Unknown alg {alg}')

        # erase lower (upper?) triangle of kq
        mask = torch.tril(torch.ones(n_toks, n_toks), diagonal=1).to(device)
        kq = kq * mask.reshape(n_toks, n_toks, 1)

        n_toks1 = n_toks - 1
        kq = kq[1:,1:,:]
        a = a[1:,1:].reshape(n_toks1, n_toks1, 1)
        reducer = TruncatedSVD(n_components=3)
        kqat = torch.nn.functional.normalize(torch.tensor(reducer.fit_transform(kq.reshape(n_toks1 * n_toks1, dkq))))
        colors = np.clip(torch.abs(kqat.reshape(n_toks1, n_toks1, 3)) * (a ** 0.3), 0, 1)
        the_ax = ax[layer - min_layer, head - min_head]
        the_ax.imshow(colors, interpolation='nearest')
        the_ax.set_xticks([])
        the_ax.set_yticks([])
        the_ax.set_title(f'L{layer} H{head}')
plt.show()
