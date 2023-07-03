import numpy as np
import transformer_lens
import torch
from matplotlib import pyplot as plt

torch.set_grad_enabled(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = transformer_lens.HookedTransformer.from_pretrained('gpt2-small', device=device)
print(f'n_layers = {model.cfg.n_layers}, n_heads = {model.cfg.n_heads}, d_model = {model.cfg.d_model}')

dependencies_k = np.zeros((model.cfg.n_layers * model.cfg.n_heads, model.cfg.n_layers * model.cfg.n_heads), dtype=np.float32)
dependencies_q = np.zeros((model.cfg.n_layers * model.cfg.n_heads, model.cfg.n_layers * model.cfg.n_heads), dtype=np.float32)
dependencies_v = np.zeros((model.cfg.n_layers * model.cfg.n_heads, model.cfg.n_layers * model.cfg.n_heads), dtype=np.float32)
for l0 in range(model.cfg.n_layers):
    print(l0)
    for h0 in range(model.cfg.n_heads):
        for l1 in range(l0 + 1, model.cfg.n_layers):
            for h1 in range(model.cfg.n_heads):
                v0 = model.blocks[l0].attn.W_V[h0,:,:]
                o0 = model.blocks[l0].attn.W_O[h0,:,:]
                k1 = model.blocks[l1].attn.W_K[h1,:,:]
                q1 = model.blocks[l1].attn.W_Q[h1,:,:]
                v1 = model.blocks[l1].attn.W_V[h1,:,:]

                m0 = torch.nn.functional.normalize(v0 @ o0, dim=0)
                mk1 = torch.nn.functional.normalize(k1, dim=1)
                mq1 = torch.nn.functional.normalize(q1, dim=1)
                mv1 = torch.nn.functional.normalize(v1, dim=1)

                d_k = torch.linalg.matrix_norm(m0 @ mk1)
                d_q = torch.linalg.matrix_norm(m0 @ mq1)
                d_v = torch.linalg.matrix_norm(m0 @ mv1)
                dependencies_k[l0 * model.cfg.n_heads + h0, l1 * model.cfg.n_heads + h1] = d_k
                dependencies_q[l0 * model.cfg.n_heads + h0, l1 * model.cfg.n_heads + h1] = d_q
                dependencies_v[l0 * model.cfg.n_heads + h0, l1 * model.cfg.n_heads + h1] = d_v

fig, ax = plt.subplots(1, 3)
ax[0].imshow(dependencies_k)
ax[1].imshow(dependencies_q)
ax[2].imshow(dependencies_v)
plt.show()
