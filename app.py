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

def _extract_flow(model, cache, layer, head, tok, n_tokens):
    attn = cache[f'blocks.{layer}.attn.hook_pattern'][0,head,tok,:]
    wv = model.blocks[layer].attn.W_V.data[head,:,:]
    l2 = layer - 1
    if l2 < 0:
        raise ValueError('Cannot extract flow from layer 0')
    mlp_out = cache[f'blocks.{l2}.hook_mlp_out'][0,:,:]
    mlp_residual = cache[f'blocks.{l2}.hook_resid_post'][0,:,:]
    stuff = (mlp_out @ wv)
    residual_stuff = (mlp_residual @ wv)
    mlp_contrib = (1 - stuff.norm(dim=1) / residual_stuff.norm(dim=1)) * attn

    result = torch.zeros((n_tokens,2))
    result[:,0] = mlp_contrib
    result[:,1] = attn

    return result

def _extract_from_cache(model, cache, n_tokens, name: str, dims: list[int], dtype: str) -> bytearray:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    if name == 'attention' and dtype == 'float32':
        t = torch.zeros((n_layers, n_heads, n_tokens, n_tokens), dtype=torch.float32)
        for layer in range(n_layers):
            t[layer,:,:,:] = cache[f'blocks.{layer}.attn.hook_pattern']
    elif name == 'flow' and dtype == 'float32':
        t = torch.zeros((n_tokens,n_tokens,2), dtype=torch.float32)
        for tok in range(n_tokens):
            layer = 4
            head = 0
            t[tok,:,:] = _extract_flow(model, cache, layer, head, tok, n_tokens)
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
