function _handle_error(e) {
    console.log(e);
}

function _draw_transformer(model) {
    const svg = document.getElementById('svg');
    svg.innerHTML = '';
    const width = svg.clientWidth;
    const height = svg.clientHeight;
    const margin = 5;

    const residual_stream = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    residual_stream.setAttribute('x1', margin);
    residual_stream.setAttribute('y1', height - margin);
    residual_stream.setAttribute('x2', margin);
    residual_stream.setAttribute('y2', margin);
    residual_stream.setAttribute('stroke', 'black');

    const box_space = (height - 2 * margin) / model.n_layers;
    const mlp_height = box_space * 0.3 - 2 * margin;
    const mlp_width = mlp_height * 3;
    const box_size = box_space * 0.5;
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
            attn_box.setAttribute('fill', 'none');
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
        _handle_error("Error from server");
        return;
    }
    const data = await response.json();
    console.log(data);
    _draw_transformer(data.model);
}