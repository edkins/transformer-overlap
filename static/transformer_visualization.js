function _handle_error(e) {
    console.log(e);
}

function _draw_transformer(model) {
    const svg = document.getElementById('svg');
    svg.innerHTML = '';
    const width = svg.clientWidth;
    const height = svg.clientHeight;
    const margin = 5;
    const box_space = (height - 2 * margin) / model.n_layers;
    const mlp_height = box_space * 0.3 - 2 * margin;
    const mlp_width = mlp_height * 3;
    const box_size = box_space * 0.5;

    const residual_stream = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    residual_stream.setAttribute('x1', margin);
    residual_stream.setAttribute('y1', height - margin);
    residual_stream.setAttribute('x2', margin);
    residual_stream.setAttribute('y2', margin);
    residual_stream.setAttribute('stroke', 'black');

    for (let layer = 0; layer < model.n_layers; layer++) {
        const resid_broadcast = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        resid_broadcast.setAttribute('x1', margin);
        resid_broadcast.setAttribute('y1', layer * box_space + box_size + 5 * margin + mlp_height);
        resid_broadcast.setAttribute('x2', (model.n_heads - 1) * box_space + box_size / 2 + 2 * margin);
        resid_broadcast.setAttribute('y2', layer * box_space + box_size + 5 * margin + mlp_height);
        resid_broadcast.setAttribute('stroke', 'black');
        svg.appendChild(resid_broadcast);

        for (let head = 0; head < model.n_heads; head++) {
            const resid_in = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            resid_in.setAttribute('x1', head * box_space + box_size / 2 + 2 * margin);
            resid_in.setAttribute('y1', layer * box_space + box_size + 5 * margin + mlp_height);
            resid_in.setAttribute('x2', head * box_space + box_size / 2 + 2 * margin);
            resid_in.setAttribute('y2', layer * box_space + box_size + 4 * margin + mlp_height);
            resid_in.setAttribute('stroke', 'black');
            svg.appendChild(resid_in);

            const attn_box = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            attn_box.setAttribute('x', head * box_space + 2 * margin);
            attn_box.setAttribute('y', layer * box_space + 4 * margin + mlp_height);
            attn_box.setAttribute('width', box_size);
            attn_box.setAttribute('height', box_size);
            attn_box.setAttribute('stroke', 'black');
            attn_box.setAttribute('fill', '#888');
            svg.appendChild(attn_box);

            const resid_out = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            resid_out.setAttribute('x1', head * box_space + box_size / 2 + 2 * margin);
            resid_out.setAttribute('y1', layer * box_space + 4 * margin + mlp_height);
            resid_out.setAttribute('x2', head * box_space + box_size / 2 + 2 * margin);
            resid_out.setAttribute('y2', layer * box_space + 3 * margin + mlp_height);
            resid_out.setAttribute('stroke', 'black');
            svg.appendChild(resid_out);
        }

        const resid_collect = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        resid_collect.setAttribute('x1', (model.n_heads - 1) * box_space + box_size / 2 + 2 * margin);
        resid_collect.setAttribute('y1', layer * box_space + mlp_height + 3 * margin);
        resid_collect.setAttribute('x2', margin);
        resid_collect.setAttribute('y2', layer * box_space + mlp_height + 3 * margin);
        resid_collect.setAttribute('stroke', 'black');
        svg.appendChild(resid_collect);

        const mlp_in = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        mlp_in.setAttribute('x1', margin);
        mlp_in.setAttribute('y1', layer * box_space + 2 * margin + mlp_height);
        mlp_in.setAttribute('x2', margin * 3);
        mlp_in.setAttribute('y2', layer * box_space + 2 * margin + mlp_height);
        mlp_in.setAttribute('stroke', 'black');
        svg.appendChild(mlp_in);

        const mlp_box = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        mlp_box.setAttribute('x', margin * 3);
        mlp_box.setAttribute('y', layer * box_space + 2 * margin);
        mlp_box.setAttribute('width', margin * 3 + mlp_width);
        mlp_box.setAttribute('height', mlp_height);
        mlp_box.setAttribute('stroke', 'black');
        mlp_box.setAttribute('fill', 'white');
        svg.appendChild(mlp_box);

        const mlp_out = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        mlp_out.setAttribute('x1', margin * 3);
        mlp_out.setAttribute('y1', layer * box_space + 2 * margin);
        mlp_out.setAttribute('x2', margin);
        mlp_out.setAttribute('y2', layer * box_space + 2 * margin);
        mlp_out.setAttribute('stroke', 'black');
        svg.appendChild(mlp_out);
    }

    svg.appendChild(residual_stream);
}

function _draw_attention(n_layers, n_heads, n_tokens, dataview) {
    const svg = document.getElementById('svg');
    const height = svg.clientHeight;
    const margin = 5;
    const box_space = (height - 2 * margin) / n_layers;
    const mlp_height = box_space * 0.3 - 2 * margin;
    const box_size = box_space * 0.5;

    for (let layer = 0; layer < n_layers; layer++) {
        for (let head = 0; head < n_heads; head++) {
            for (let token0 = 0; token0 < n_tokens; token0++) {
                for (let token1 = 0; token1 <= token0; token1++) {
                    const ix = ((((layer * n_heads) + head) * n_tokens) + token0) * n_tokens + token1;
                    const value = dataview.getFloat32(ix * 4, true);
                    const r = Math.min(Math.floor(1000 * value), 255);
                    const g = Math.min(Math.floor(300 * value), 255);
                    const b = Math.min(Math.floor(100 * value), 255);
                    const color = `rgb(${r}, ${g}, ${b})`;
                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('x', head * box_space + 2 * margin + token1 * (box_size / n_tokens));
                    rect.setAttribute('y', layer * box_space + 4 * margin + mlp_height + (n_tokens - 1 - token0) * (box_size / n_tokens));
                    rect.setAttribute('width', box_size / n_tokens);
                    rect.setAttribute('height', box_size / n_tokens);
                    rect.setAttribute('fill', color);
                    svg.appendChild(rect);
                }
            }
        }
    }
}

async function transformer_visualization() {
    const prompt_text = document.getElementById('prompt').value;
    const model_name = document.getElementById('model').value;
    const response = await fetch('/api/tokenize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt_text,
            model_name,
        })
    });
    if (!response.ok) {
        _handle_error("Error from server when tokenizing prompt.");
        return;
    }
    const prompt_data = await response.json();

    const n_tokens = prompt_data.prompt.tokens.length;

    let byte_offset = 0;
    const sections = [];
    sections.push({
        name: 'attention',
        byte_offset,
        byte_length: prompt_data.model.n_layers * prompt_data.model.n_heads * n_tokens * n_tokens * 4,
        dims: [prompt_data.model.n_layers, prompt_data.model.n_heads, n_tokens, n_tokens],
        dtype: 'float32',
    });
    byte_offset += sections[0].byte_length;

    const response2 = await fetch('/api/activations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt_text,
            model_name,
            byte_length: byte_offset,
            sections,
        })
    });
    if (!response2.ok) {
        _handle_error("Error from server with activations");
        return;
    }
    const result_byte_array = await response2.arrayBuffer();
    const activation_dataview = new DataView(result_byte_array, sections[0].byte_offset, sections[0].byte_length);
    _draw_transformer(prompt_data.model);
    _draw_attention(prompt_data.model.n_layers, prompt_data.model.n_heads, n_tokens, activation_dataview);
}