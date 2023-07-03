import numpy as np
import transformer_lens
import torch
from matplotlib import pyplot as plt

prompt = 'The quick brown fox jumps over the lazy dog.'

torch.set_grad_enabled(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = transformer_lens.HookedTransformer.from_pretrained('gpt2-small', device=device)
model.set_use_attn_result(True)   # says it can burn through gpu memory
n_heads = model.cfg.n_heads
print(f'n_layers = {model.cfg.n_layers}, n_heads = {n_heads}, d_model = {model.cfg.d_model}')

toks = model.to_tokens(prompt)[0]
n_toks = len(toks)
print(f'prompt = {prompt}')
print(f'n_toks = {n_toks}')

predictions, cache = model.run_with_cache(prompt)

fig, ax = plt.subplots(1, 1)
viz = np.zeros((12 * n_heads, 12 * n_heads, 3))
for llayer in range(1, 12):
    print(f'layer {llayer}')
    for lhead in range(n_heads):
        for channel, letter in enumerate(['k', 'q', 'v']):
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

            attn_contrib = []
            attns_contrib = []
            mlp_contrib = []
            for layer in range(llayer):
                a = cache[f'blocks.{layer}.hook_attn_out'][0]
                m = cache[f'blocks.{layer}.hook_mlp_out'][0]
                r = cache[f'blocks.{layer}.attn.hook_result'][0] + model.blocks[layer].attn.b_O / n_heads
                attn_contrib.append((a @ inp_stuff) * ln1_scale)
                mlp_contrib.append((m @ inp_stuff) * ln1_scale)
                rc = torch.zeros((n_heads, n_toks, 64), device=device)
                for head in range(n_heads):
                    rc[head,:,:] = (r[:,head,:] @ inp_stuff) * ln1_scale
                attns_contrib.append(rc)

            debug_sum_start = (resid_start @ inp_stuff) * ln1_scale + inp_bias
            debug_sum = debug_sum_start
            for layer in range(llayer):
                for head in range(n_heads):
                    debug_sum += attns_contrib[layer][head,:,:]
                debug_sum += mlp_contrib[layer]

            for layer in range(llayer):
                for head in range(n_heads):
                    x = layer * n_heads + head
                    y = llayer * n_heads + lhead
                    viz[x,y,channel] = attns_contrib[layer][head,:,:].norm(dim=1).cpu().numpy().mean(axis=0)

ax.imshow((viz ** 0.5) * 0.5)
plt.show()
