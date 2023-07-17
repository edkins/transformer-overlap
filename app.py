from flask import Flask, redirect, request, Response
from jsonschema import validate
import torch
import transformer_lens

app = Flask(__name__)

model_cache = {}
supported_models = {'gpt2-small'}

torch.set_grad_enabled(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def _get_model(model_name):
    global model_cache, device, supported_models

    if model_name not in supported_models:
        raise ValueError(f'Model {model_name} not supported.')
    if model_name not in model_cache:
        print(f'Loading model {model_name}')
        model_cache[model_name] = transformer_lens.HookedTransformer.from_pretrained('gpt2-small', device=device)
        model_cache[model_name].set_use_attn_result(True)   # says it can burn through gpu memory
    return model_cache[model_name]

_get_model('gpt2-small')

@app.route('/')
def index():
    return redirect('static/index.html')

schema_tokenize = {
    'type': 'object',
    'properties': {
        'prompt_text': {'type': 'string'},
        'model_name': {'type': 'string'},
    },
    'required': ['prompt_text', 'model_name'],
}

@app.post('/api/tokenize')
def tokenize():
    j = request.json
    validate(instance=j, schema=schema_tokenize)
    text = j['prompt_text']
    model_name = j['model_name']
    model = _get_model(model_name)
    tokens = model.to_tokens(text)[0]
    labels = [model.tokenizer.decode(t) for t in tokens]
    return {
        'model': {
            'name': model_name,
            'n_layers': model.cfg.n_layers,
            'n_heads': model.cfg.n_heads,
            'd_model': model.cfg.d_model,
            'd_head': model.cfg.d_head,
        },
        'prompt': {
            'text': text,
            'tokens': [t.item() for t in tokens],
            'labels': labels,
        }
    }

schema_activations = {
    'type': 'object',
    'properties': {
        'prompt_text': {'type': 'string'},
        'model_name': {'type': 'string'},
        'byte_length': {'type': 'integer', 'minimum': 0},
        'sections': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'byte_offset': {'type': 'integer', 'minimum': 0},
                    'byte_length': {'type': 'integer', 'minimum': 0},
                    'dims': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}},
                    'dtype': {'enum': ['float32']},
                },
                'required': ['name', 'byte_offset', 'byte_length', 'dims', 'dtype'],
            },
        },
    },
    'required': ['prompt_text', 'model_name', 'byte_length', 'sections'],
}

def _extract_flow(model, cache, layer, n_tokens):
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head
    result = torch.zeros((n_tokens, layer, 1 + n_heads, n_tokens), dtype=torch.float32)

    final_resid = cache[f'blocks.{layer}.hook_resid_post'][0,:,:]   # token x d_model
    multiplier = final_resid * (final_resid.norm(dim=1, keepdim=True) ** -2)   # token x d_model
    multiplier_broadcast = multiplier.reshape((n_tokens, 1, d_model, 1))   # token x (head) x d_model x (token)

    for l2 in range(layer):
        v_out = cache[f'blocks.{l2}.attn.hook_v'][0,:,:,:]   # token x head x d_head
        attn = cache[f'blocks.{l2}.attn.hook_pattern'][0,:,:,:]   # head x token x token
        w_o = model.blocks[l2].attn.W_O.data   # head x d_head x d_model
        head_out = torch.einsum('thv,htu,hvm->thmu', v_out, attn, w_o)   # token x head x d_model x token
#        print(head_out.shape, multiplier_broadcast.shape)
        result[:,l2,:n_heads,:] = (head_out * multiplier_broadcast).sum(dim=2)   # token x head x token

        mlp_out = cache[f'blocks.{l2}.hook_mlp_out'][0,:,:]    # token x d_model
        mlp_vec = (mlp_out * multiplier).sum(dim=1)    # token
        for t in range(n_tokens):
            result[t,l2,n_heads,t] = mlp_vec[t]

    return result

def _extract_from_cache(model, cache, n_tokens, name: str, dims: list[int], dtype: str) -> bytearray:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    if name == 'attention' and dtype == 'float32':
        t = torch.zeros((n_layers, n_heads, n_tokens, n_tokens), dtype=torch.float32)
        for layer in range(n_layers):
            t[layer,:,:,:] = cache[f'blocks.{layer}.attn.hook_pattern']
    elif name == 'flow3' and dtype == 'float32':
        t = _extract_flow(model, cache, 3, n_tokens)
    else:
        raise ValueError(f'Unsupported name/dtype combination: {name}/{dtype}')

    if tuple(t.shape) != tuple(dims):
        raise ValueError(f'Expected dims {dims}, got {t.shape}')

    return t.numpy().tobytes()

@app.post('/api/activations')
def get_activations():
    j = request.json
    validate(instance=j, schema=schema_activations)
    if j['byte_length'] > 1_000_000_000:
        raise ValueError("byte_length is too huge")
    text = j['prompt_text']
    model = _get_model(j['model_name'])
    result = bytearray(j['byte_length'])
    n_tokens = model.to_tokens(text).shape[1]
    _, cache = model.run_with_cache(text)
    for section in j['sections']:
        name = section['name']
        byte_offset = section['byte_offset']
        byte_length = section['byte_length']
        dims = section['dims']
        dtype = section['dtype']
        if byte_offset + byte_length > j['byte_length']:
            raise ValueError("byte_length is too small. Section won't fit.")
        chunk = _extract_from_cache(model, cache, n_tokens, name, dims, dtype)
        if len(chunk) != byte_length:
            raise ValueError(f'Expected byte_length {byte_length}, got {len(chunk)}')
        result[byte_offset:byte_offset+byte_length] = chunk
    return Response(bytes(result), mimetype='application/octet-stream')
