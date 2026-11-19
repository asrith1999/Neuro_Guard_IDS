"""Render an offline animation of one REAL trained MLP forward pass.

The visual animation is a teaching aid: neurons and predictions are computed
from the saved scikit-learn pipeline; display timing is illustrative, not the
wall-clock speed of inference or a training/backpropagation animation.
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from core import check_features


LAYER_X = (220, 474, 732, 970)
TOP_Y, BOTTOM_Y = 98, 570
SVG_WIDTH, SVG_HEIGHT = 1100, 645


def inspect_forward_pass(model, row: pd.DataFrame) -> dict:
    """Compute the encoded input, both ReLU layers, and sigmoid output."""
    validated = check_features(row)
    if len(validated) != 1:
        raise ValueError("Choose exactly one network-flow row to animate.")
    transformer = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    if classifier.activation != "relu" or classifier.out_activation_ != "logistic":
        raise ValueError("This visualizer needs the project's ReLU + logistic MLP architecture.")
    encoded = transformer.transform(validated)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    signal = np.asarray(encoded, dtype=float)[0]
    layers = [signal]
    for weight_matrix, bias in zip(classifier.coefs_[:-1], classifier.intercepts_[:-1]):
        signal = np.maximum(signal @ weight_matrix + bias, 0.0)
        layers.append(signal)
    out_logit = float((signal @ classifier.coefs_[-1] + classifier.intercepts_[-1]).item())
    out_logit = float(np.clip(out_logit, -700, 700))
    suspicious = float(1.0 / (1.0 + np.exp(-out_logit)))
    model_score = float(model.predict_proba(validated)[0, list(model.classes_).index(1)])
    if abs(suspicious - model_score) > 1e-7:
        raise ValueError("The calculated activations do not match the trained model's output.")
    layers.append(np.array([suspicious]))
    return {
        "layers": layers,
        "weights": classifier.coefs_,
        "feature_names": [
            name.replace("numeric__", "").replace("protocol__protocol_", "Protocol ")
            for name in transformer.get_feature_names_out()
        ],
        "suspicious_score": suspicious,
        "label": "Suspicious" if suspicious >= 0.5 else "Normal",
    }


def _pos(index: int, count: int) -> float:
    if count == 1:
        return 336.0
    return TOP_Y + (BOTTOM_Y - TOP_Y) * index / (count - 1)


def _node_strength(values: np.ndarray, index: int, layer: int) -> float:
    if layer == 3:
        return 0.55 + 0.45 * float(values[index])
    v = float(values[index])
    if layer == 0:
        return 0.2 + 0.8 * float(np.tanh(abs(v) / 2))
    largest = max(float(np.max(values)), 1e-10)
    return 0.13 + 0.87 * (max(0.0, v) / largest) ** 0.6


def render_animation_html(data: dict) -> str:
    """Self-contained HTML: all drawing, motion, and controls work offline."""
    layers = data["layers"]
    counts = [len(a) for a in layers]
    if counts[1:] != [32, 16, 1]:
        raise ValueError("Unexpected neural network architecture.")
    connections = []
    for transition, weights in enumerate(data["weights"]):
        prev = layers[transition]
        for dest in range(weights.shape[1]):
            contributions = prev * weights[:, dest]
            cap = 4 if transition == 0 else (6 if transition == 1 else 16)
            strongest = np.argsort(np.abs(contributions))[-cap:]
            for src in strongest:
                value = float(contributions[src])
                opacity = 0.13 + min(0.38, abs(value) / (max(float(np.max(np.abs(contributions))), 1e-9)) * 0.36)
                color = "#29cde0" if value >= 0 else "#fbad59"
                x1, x2 = LAYER_X[transition], LAYER_X[transition + 1]
                y1, y2 = _pos(int(src), counts[transition]), _pos(dest, counts[transition + 1])
                connections.append(
                    f'<line class="connection" x1="{x1}" y1="{y1:.2f}" x2="{x2}" y2="{y2:.2f}" '
                    f'stroke="{color}" stroke-opacity="{opacity:.3f}"/>'
                    f'<line class="signal step-{transition + 1}" x1="{x1}" y1="{y1:.2f}" x2="{x2}" y2="{y2:.2f}" '
                    f'stroke="{color}"/>'
                )
    dots = []
    for layer, values in enumerate(layers):
        x = LAYER_X[layer]
        for i, value in enumerate(values):
            y = _pos(i, len(values))
            brightness = _node_strength(values, i, layer)
            r = 9.3 if layer == 0 else (6.0 if layer == 1 else (8.1 if layer == 2 else 19))
            color = "#ffb35c" if layer == 0 and float(value) < 0 else "#37e5ea"
            if layer == 3:
                color = "#ffa55d" if data["label"] == "Suspicious" else "#5df3b0"
            title = (
                f"{data['feature_names'][i]} = {float(value):.4f}"
                if layer == 0 else f"Layer {layer} · neuron {i + 1} = {float(value):.5f}"
            )
            dots.append(
                f'<circle class="neuron layer-{layer}" cx="{x}" cy="{y:.2f}" r="{r}" '
                f'fill="{color}" fill-opacity="{0.16 + 0.78 * brightness:.3f}" '
                f'stroke="{color}" stroke-width="1.4"><title>{html.escape(title)}</title></circle>'
            )
            if layer == 0:
                label = html.escape(data["feature_names"][i])
                dots.append(f'<text class="input-name" x="199" y="{y+4:.2f}" text-anchor="end">{label}</text>')
    headers = []
    for idx, (title, detail) in enumerate((
        ("INPUT", f"{counts[0]} features"),
        ("HIDDEN 1", "32 neurons • ReLU"),
        ("HIDDEN 2", "16 neurons • ReLU"),
        ("OUTPUT", "Sigmoid")
    )):
        x = LAYER_X[idx]
        headers.append(
            f'<text class="layer-title" x="{x}" y="29" text-anchor="middle">{title}</text>'
            f'<text class="layer-detail" x="{x}" y="49" text-anchor="middle">{html.escape(detail)}</text>'
        )
    score = float(data["suspicious_score"])
    result_label = html.escape(data["label"])
    svg = (
        f'<svg viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" '
        'aria-label="Animated neural network with encoded inputs, 32 and 16 hidden neurons, and one sigmoid output">'
        '<rect x="0" y="0" width="1100" height="645" rx="20" fill="#071a2b"/>'
        '<g>' + ''.join(headers) + '</g>'
        '<g>' + ''.join(connections) + '</g>'
        '<g>' + ''.join(dots) + '</g>'
        f'<text class="output-percent" x="970" y="389" text-anchor="middle">{score:.1%} suspicious score</text>'
        '<text class="footnote" x="210" y="624">Teal: positive contribution • Orange: negative contribution • Hover over a neuron for its actual value</text>'
        '</svg>'
    )
    # No remote scripts or styles: the preview runs offline by double-clicking it.
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuroGuard | Neural Forward Pass Animation</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;font:14px system-ui,-apple-system,Segoe UI,sans-serif;color:#dceefa;background:#061321}}
.wrap{{padding:14px 18px 18px;max-width:1400px;margin:auto}}
.top{{display:flex;gap:12px;flex-wrap:wrap;align-items:center;justify-content:space-between;margin-bottom:11px}}
.heading{{font-size:18px;font-weight:750;letter-spacing:.1px}}
.sub{{font-size:12px;color:#9fbed1;margin-top:3px}}
.controls{{display:flex;align-items:center;flex-wrap:wrap;gap:7px}}
button,select{{border:1px solid #33546b;border-radius:9px;padding:8px 12px;color:#e3f4ff;background:#102f46;font:inherit;cursor:pointer}}
button:hover{{background:#195371}}
.info{{display:flex;justify-content:space-between;gap:10px;align-items:center;background:#10304a;border:1px solid #2b5371;border-radius:10px;padding:10px 12px;margin:10px 0;min-height:45px}}
#phase{{color:#aef5fb;font-weight:680}}
#stepInfo{{color:#c3d5e5;font-size:12px}}
svg{{display:block;width:100%;height:auto;border:1px solid #29495f;border-radius:15px}}
.layer-title{{fill:#e9f7ff;font-weight:760;font-size:14px;letter-spacing:1.2px}}
.layer-detail{{fill:#8bb7d0;font-size:11px}}
.input-name{{fill:#d3e8f5;font-size:10.4px}}
.footnote{{fill:#9bbbd0;font-size:11.3px}}
.output-percent{{fill:#d7effa;font-size:12px}}
.connection{{stroke-width:1}}
.signal{{stroke-width:2;stroke-opacity:.87;stroke-dasharray:5 15;stroke-dashoffset:0;opacity:0;animation:trickle .55s linear infinite}}
.neuron{{transition:filter .22s,stroke-width .22s}}
.network[data-step="0"] .layer-0,.network[data-step="1"] .layer-1,
.network[data-step="2"] .layer-2,.network[data-step="3"] .layer-3{{filter:drop-shadow(0 0 7px #48e7ef);stroke-width:3}}
.network[data-step="1"] .step-1,.network[data-step="2"] .step-2,
.network[data-step="3"] .step-3{{opacity:.9}}
@keyframes trickle{{to{{stroke-dashoffset:-40}}}}
.below{{color:#abc6d7;font-size:12px;margin:9px 0 0;line-height:1.5}}
@media(prefers-reduced-motion:reduce){{.signal{{animation:none}}.neuron{{transition:none}}}}
@media(max-width:640px){{.info{{display:block}} #stepInfo{{margin-top:4px}}}}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div><div class="heading">🧠 Neural Network — Animated Forward Pass</div>
    <div class="sub">Trained model activations, not a random drawing · {result_label} · score {score:.1%}</div></div>
    <div class="controls"><button id="play">⏸ Pause</button><button id="replay">↺ Replay</button>
    <label for="speed">Speed</label><select id="speed"><option value="1700">Slow</option><option value="1100" selected>Normal</option><option value="550">Fast</option></select></div>
  </div>
  <div class="network" id="network" data-step="0">{svg}</div>
  <div class="info"><div id="phase">Step 1 / 4 · Input signals</div><div id="stepInfo">The nine CSV fields become eleven encoded numbers after preprocessing.</div></div>
  <div class="below">The network has <strong>11 → 32 → 16 → 1</strong> computing units. The moving lines show a selected subset of strong weighted contributions; not every connection is drawn. The output color reflects the model score. This animation replays one inference, not training or live traffic.</div>
</div>
<script>
(function() {{
 const network=document.getElementById('network'), phase=document.getElementById('phase');
 const detail=document.getElementById('stepInfo');
 const play=document.getElementById('play'), replay=document.getElementById('replay');
 const speed=document.getElementById('speed');
 let step=0, paused=false, timer=null;
 const titles=['Step 1 / 4 · Input signals','Step 2 / 4 · Hidden layer 1','Step 3 / 4 · Hidden layer 2','Step 4 / 4 · Final prediction'];
 const details=[
  'Validated CSV fields are normalized and the protocol is one-hot encoded.',
  '32 neurons combine encoded numbers using learned weights, biases, and ReLU.',
  '16 neurons transform the previous layer’s outputs using ReLU.',
  'A sigmoid produces a {score:.1%} suspicious-class score: {result_label}.'
 ];
 function show() {{ network.dataset.step=String(step); phase.textContent=titles[step]; detail.textContent=details[step]; }}
 function stop() {{ if(timer!==null) clearInterval(timer); timer=null; }}
 function run() {{ stop(); if(!paused) timer=setInterval(()=>{{step=(step+1)%4;show();}}, Number(speed.value)); }}
 play.addEventListener('click',()=>{{paused=!paused;play.textContent=paused?'▶ Play':'⏸ Pause';run();}});
 replay.addEventListener('click',()=>{{step=0;paused=false;play.textContent='⏸ Pause';show();run();}});
 speed.addEventListener('change',run);
 document.addEventListener('visibilitychange',()=>{{if(document.hidden)stop();else run();}});
 show();run();
}})();
</script></body></html>'''


def write_offline_preview(model, sample_path: Path, destination: Path) -> Path:
    """Generate an opening-in-browser preview using a real example and trained model."""
    frame = pd.read_csv(sample_path).head(1)
    result = inspect_forward_pass(model, frame)
    destination.write_text(render_animation_html(result), encoding="utf-8")
    return destination
