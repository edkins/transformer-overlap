"use strict";

function _handle_error(e) {
    console.log(e);
}

let flow_data = undefined;

function mouse_over_token(e) {
    const token_index = parseInt(e.target.dataset.token_index);
    const token_row = parseInt(e.target.dataset.row);
    const n_tokens = flow_data.n_tokens;
    for (let row = 0; row < 3; row++) {
        for (let i = 0; i < n_tokens; i++) {
            const id = `row_${row}_token_${i}`;
            const rect = document.getElementById(id);
            let value = 0;
            if (row < flow_data.flow[token_index][i].length) {
                value = flow_data.flow[token_index][i][row];
            } else if (i === token_index) {
                value = 1;
            }
            let r,g,b;
            if (value >= 0) {
                r = Math.floor((1-value) * 255);
                g = Math.floor((1-0.8 * value) * 255);
                b = Math.floor((1-0.5 * value) * 255);
            } else {
                r = 255;
                g = Math.floor((1+value) * 255);
                b = Math.floor((1+value) * 255);
            }
            const color = `rgb(${r}, ${g}, ${b})`;
            rect.setAttribute('fill', color);
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
    const layer = 4;
    const row_labels = [
        `MLP ${layer-1}`,
        `Attn ${layer}`,
        '',
    ]
    for (let row = 0; row < 3; row++) {
        for (let i = 0; i < n_tokens; i++) {
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', i * label_width);
            rect.setAttribute('y', 200 - row * label_height);
            rect.setAttribute('width', label_width);
            rect.setAttribute('height', label_height);
            rect.setAttribute('fill', 'white');
            rect.setAttribute('stroke', 'black');
            rect.setAttribute('rx', 3);
            rect.setAttribute('ry', 3);
            rect.addEventListener('mouseover', mouse_over_token);
            rect.id = `row_${row}_token_${i}`;
            rect.dataset.token_index = i;
            rect.dataset.row = row;
            svg.appendChild(rect);

            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', 2 + i * label_width);
            label.setAttribute('y', 200 - (row - 0.5) * label_height);
            label.setAttribute('font-size', 10);
            label.setAttribute('font-family', 'monospace');
            label.setAttribute('dominant-baseline', 'middle');
            label.textContent = labels[i];
            svg.appendChild(label);
        }
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', 2 + n_tokens * label_width);
        label.setAttribute('y', 200 - (row - 0.5) * label_height);
        label.setAttribute('font-size', 10);
        label.setAttribute('font-family', 'sans-serif');
        label.setAttribute('dominant-baseline', 'middle');
        label.textContent = row_labels[row];
        svg.appendChild(label);
    }
}

function unpack_flow_dataview(dataview, n_tokens) {
    const result = [];
    for (let i = 0; i < n_tokens; i++) {
        const row = [];
        for (let j = 0; j < n_tokens; j++) {
            const flow = [];
            flow.push(dataview.getFloat32((i * n_tokens + j) * 2 * 4, true));
            flow.push(dataview.getFloat32((i * n_tokens + j) * 2 * 4 + 4, true));
            row.push(flow);
        }
        result.push(row);
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

    const n_tokens = prompt_data.prompt.tokens.length;

    let byte_offset = 0;
    const sections = [];
    sections.push({
        name: 'flow',
        byte_offset,
        byte_length: n_tokens * n_tokens * 2 * 4,
        dims: [n_tokens, n_tokens, 2],
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
        flow: unpack_flow_dataview(dataview, n_tokens),
    };

    draw_flow(prompt_data.prompt.labels);
}