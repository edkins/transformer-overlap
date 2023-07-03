from flask import Flask, redirect, request
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

@app.route('/')
def index():
    return redirect('static/index.html')

@app.post('/api/tokenize')
def put_document():
    j = request.json
    text = j['prompt_text']
    if not isinstance(text, str):
        raise ValueError(f'text must be a string, not {type(text)}')
    model_name = j['model_name']
    model = _get_model(model_name)
    tokens = model.to_tokens(text)[0]
    return {
        'model': {
            'name': model_name,
            'n_layers': model.cfg.n_layers,
            'n_heads': model.cfg.n_heads,
            'd_model': model.cfg.d_model,
        },
        'prompt': {
            'text': text,
            'tokens': [t.item() for t in tokens],
        }
    }