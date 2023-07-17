"use strict";

function _handle_error(e) {
    console.log(e);
}

let flow_data = undefined;

function mouse_over_token(e) {
    const token_index = parseInt(e.target.dataset.token_index);
    const n_tokens = flow_data.n_tokens;
    for (let layer = 0; layer < flow_data.layer + 1; layer++) {
        for (let head = 0; head < flow_data.n_heads + 1; head++) {
            for (let i = 0; i < n_tokens; i++) {
                const id = `layer_${layer}_head_${head}_token_${i}`;
                const rect = document.getElementById(id);
                let value = flow_data.flow[token_index][layer][head][i];
                let r,g,b;
                if (value >= 0) {
                    const v = Math.pow(value, 0.5);
                    r = Math.floor((1-v) * 255);
                    g = Math.floor((1-0.8 * v) * 255);
                    b = Math.floor((1-0.5 * v) * 255);
                } else {
                    const v = Math.pow(-value, 0.5);
                    r = 255;
                    g = Math.floor((1-v) * 255);
                    b = Math.floor((1-v) * 255);
                }
                const color = `rgb(${r}, ${g}, ${b})`;
                rect.setAttribute('fill', color);
            }
        }
    }
}

function draw_flow(labels) {
    const svg = document.getElementById('svg');
    svg.innerHTML = '';
    const width = svg.clientWidth;
    const height = svg.clientHeight;
    const label_width = 100;
    const label_height = 20;
    const n_tokens = labels.length;
    for (let layer = 0; layer < flow_data.layer; layer++) {
        for (let head = 0; head < flow_data.n_heads + 1; head++) {
            const row = layer * (flow_data.n_heads + 2) + head;
            for (let i = 0; i < n_tokens; i++) {
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', i * label_width);
                rect.setAttribute('y', height - row * label_height);
                rect.setAttribute('width', label_width);
                rect.setAttribute('height', label_height);
                rect.setAttribute('fill', 'white');
                rect.setAttribute('stroke', 'black');
                rect.setAttribute('rx', 3);
                rect.setAttribute('ry', 3);
                rect.addEventListener('mouseover', mouse_over_token);
                rect.id = `layer_${layer}_head_${head}_token_${i}`;
                rect.dataset.token_index = i;
                rect.dataset.layer = layer;
                rect.dataset.head = head;
                svg.appendChild(rect);

                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', 2 + i * label_width);
                label.setAttribute('y', height - (row - 0.5) * label_height);
                label.setAttribute('font-size', 10);
                label.setAttribute('font-family', 'monospace');
                label.setAttribute('dominant-baseline', 'middle');
                label.textContent = labels[i];
                svg.appendChild(label);
            }
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', 2 + n_tokens * label_width);
            label.setAttribute('y', height - (row - 0.5) * label_height);
            label.setAttribute('font-size', 10);
            label.setAttribute('font-family', 'sans-serif');
            label.setAttribute('dominant-baseline', 'middle');
            if (head < flow_data.n_heads) {
                label.textContent = `Layer ${layer} Head ${head}`;
            } else {
                label.textContent = `Layer ${layer} MLP`;
            }
            svg.appendChild(label);
        }
    }
}

function unpack_flow_dataview(dataview, final_layer, n_heads, n_tokens) {
    const result = [];
    for (let h = 0; h < n_tokens; h++) {
        const tok = [];
        for (let i = 0; i < final_layer; i++) {
            const layer = [];
            for (let j = 0; j < n_heads + 1; j++) {
                const head = []
                for (let k = 0; k < n_tokens; k++) {
                    head.push(dataview.getFloat32((((h * final_layer + i) * (n_heads + 1) + j) * n_tokens + k) * 4, true));
                }
                layer.push(head);
            }
            tok.push(layer);
        }
        result.push(tok);
    }
    return result;
}

async function flow() {
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

    const n_layers = prompt_data.model.n_layers;
    const n_heads = prompt_data.model.n_heads;
    const n_tokens = prompt_data.prompt.tokens.length;

    let byte_offset = 0;
    const layer = 3;
    const sections = [];
    sections.push({
        name: `flow${layer}`,
        byte_offset,
        byte_length: n_tokens * layer * (n_heads + 1) * n_tokens * 4,
        dims: [n_tokens, layer, n_heads + 1, n_tokens],
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
    const dataview = new DataView(result_byte_array, sections[0].byte_offset, sections[0].byte_length);

    flow_data = {
        n_tokens,
        layer,
        n_heads,
        flow: unpack_flow_dataview(dataview, layer, n_heads, n_tokens),
    };

    draw_flow(prompt_data.prompt.labels);
}