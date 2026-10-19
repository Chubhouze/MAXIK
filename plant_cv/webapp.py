"""Minimal local test UI for the plant-health pipeline: upload a photo, see the result.

    python webapp.py
"""
import base64
import hmac
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from dashboard_ui import DASHBOARD_PAGE, POINT_PAGE, dashboard_context, point_context
from maxik_cv.pipeline import PlantInspector
from maxik_cv.segment import overlay, plant_mask
from maxik_cv.tomato import OnnxTomatoHealthClassifier, PlantDetector, TomatoInspector
from maxik_cv.tomato.history import TomatoHistory
from maxik_cv.tomato.service import SENSOR_KEYS, analyze

# Anchored to this file's own directory, not the process cwd: when launched via .claude/launch.json
# the cwd is the MAXIK repo root, not plant_cv/, which previously made these relative paths silently
# not exist -> PlantInspector swallows a missing model path and falls back to rules-only with no error.
MODEL_PATH = BASE_DIR / "models_synthetic" / "plant_health.onnx"
GATE_MODEL_PATH = BASE_DIR / "models_synthetic" / "plant_gate_trained.onnx"
TOMATO_HEALTH_MODEL_PATH = BASE_DIR / "models_tomato" / "tomato_health.onnx"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # reject oversized uploads before decoding them
inspector = PlantInspector(MODEL_PATH, gate_model_path=GATE_MODEL_PATH)
tomato_health_classifier = (OnnxTomatoHealthClassifier(TOMATO_HEALTH_MODEL_PATH)
                             if TOMATO_HEALTH_MODEL_PATH.exists() else None)
tomato_inspector = TomatoInspector(PlantDetector.from_model_path(GATE_MODEL_PATH),
                                    health_classifier=tomato_health_classifier)
# Inspection history. In a container the file system is lost on redeploy unless this path is on a volume.
history = TomatoHistory(os.environ.get("MAXIK_DB_PATH", BASE_DIR / "inspections.db"))
TOMATO_MODEL_TEST_ACC = (json.loads(TOMATO_HEALTH_MODEL_PATH.with_suffix(".json").read_text(encoding="utf8")).get("test_acc", 0.0)
                         if TOMATO_HEALTH_MODEL_PATH.exists() else 0.0)

PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><title>MAXIK plant-cv test</title>
<style>
body { font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 0 16px; }
img { max-width: 420px; border: 1px solid #ccc; margin-top: 12px; }
.row { display: flex; gap: 24px; flex-wrap: wrap; }
.card { flex: 1; min-width: 300px; }
pre { background: #f4f4f4; padding: 12px; overflow-x: auto; }
.label { font-size: 1.4em; font-weight: bold; }
</style></head>
<body>
<p><a href="/tomato">Switch to tomato disease pipeline (MVP)</a></p>
<h2>MAXIK — plant health pipeline test</h2>
<form method=post enctype=multipart/form-data id="f">
  <input type=file name=photo accept="image/*" required id="file-input">
  <input type=submit value="Analyze">
</form>
<p style="color:#777;font-size:0.9em">Tip: you can also paste an image with Ctrl+V.</p>
<script>
  document.getElementById('file-input').focus();
  window.addEventListener('paste', function(e){
    var items = (e.clipboardData || window.clipboardData).items;
    if (!items) return;
    for (var i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') === 0) {
        var file = items[i].getAsFile();
        var dt = new DataTransfer();
        dt.items.add(file);
        document.getElementById('file-input').files = dt.files;
        document.getElementById('f').submit();
        break;
      }
    }
  });
</script>
{% if result %}
<hr>
<p class="label">{{ result.label }} ({{ '%.0f'|format(result.confidence*100) }}%)</p>
<p>{{ result.recommendation }}</p>
{% if result.reasons %}<p>Reasons: {{ result.reasons|join('; ') }}</p>{% endif %}
<div class="row">
  <div class="card"><b>Original</b><br><img src="data:image/jpeg;base64,{{ orig_b64 }}"></div>
  <div class="card"><b>Segmentation mask</b><br><img src="data:image/jpeg;base64,{{ mask_b64 }}"></div>
</div>
<pre>{{ result_json }}</pre>
{% endif %}
</body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method != "POST":
        return render_template_string(PAGE, result=None)

    file = request.files["photo"]
    buf = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return render_template_string(PAGE, result={"label": "error", "confidence": 0,
                                                      "recommendation": "Could not decode image.",
                                                      "reasons": []}, orig_b64="", mask_b64="", result_json="")

    report = inspector.inspect(img)
    mask = plant_mask(img)
    vis = overlay(img, mask)

    import json
    return render_template_string(
        PAGE, result=report,
        orig_b64=_to_b64(img), mask_b64=_to_b64(vis),
        result_json=json.dumps(report, indent=2, ensure_ascii=False),
    )


def _to_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


TOMATO_PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>M.A.X.I.K. — Tomato Health Check</title>
<style>
  :root {
    --green: #2e7d32; --red: #c62828; --amber: #b8620a; --gray: #6b7280;
    --ink: #1f2430; --muted: #6b7280; --line: #e6e6e0; --card: #ffffff; --bg: #f4f3ee;
    --radius: 14px;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--ink); margin: 0; padding: 32px 16px 64px;
  }
  .wrap { max-width: 640px; margin: 0 auto; }
  header { margin-bottom: 24px; }
  .brand { font-size: 0.8em; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
  h1 { font-size: 1.6em; margin: 4px 0 8px; }
  .sub { color: var(--muted); font-size: 0.95em; line-height: 1.5; margin: 0; }
  nav { margin-top: 10px; font-size: 0.9em; }
  nav a { color: var(--muted); }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 24px; margin-top: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  .dropzone {
    border: 2px dashed var(--line); border-radius: var(--radius); padding: 28px; text-align: center;
    color: var(--muted); transition: border-color .15s, background .15s;
  }
  .dropzone.drag { border-color: var(--green); background: #f1f8f1; }
  .dropzone input[type=file] { display: block; margin: 12px auto 0; }
  button {
    background: var(--ink); color: #fff; border: none; border-radius: 999px; padding: 10px 22px;
    font-size: 0.95em; cursor: pointer; margin-top: 16px;
  }
  button:hover { opacity: 0.88; }
  .status-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .status { font-size: 1.3em; font-weight: 700; margin: 0; }
  .pill { display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 0.78em;
          font-weight: 700; letter-spacing: 0.02em; color: #fff; }
  .healthy { background: var(--green); } .unhealthy { background: var(--red); }
  .uncertain { background: var(--amber); } .not_applicable { background: var(--gray); }
  .meta { color: var(--muted); font-size: 0.92em; margin: 10px 0 0; }
  .message { margin: 14px 0 0; line-height: 1.5; }
  ul.symptoms, ul.recs { padding-left: 20px; margin: 8px 0 0; line-height: 1.6; }
  .section-label { font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.06em;
                    color: var(--muted); font-weight: 700; margin: 18px 0 2px; }
  .photo { max-width: 100%; border-radius: 10px; margin-top: 16px; display: block; }
  .quality-warning { background: #fdecea; border: 1px solid #f4c4c0; color: var(--red);
                      border-radius: 10px; padding: 10px 14px; margin-top: 14px; font-size: 0.9em; }
  .placeholder-note { background: #fff8e6; border: 1px solid #f0deab; border-radius: 10px;
                       padding: 10px 14px; margin-top: 14px; font-size: 0.87em; color: #6b5300; }
  details { margin-top: 18px; }
  summary { cursor: pointer; color: var(--muted); font-size: 0.85em; }
  pre { background: #f4f4f4; padding: 12px; overflow-x: auto; border-radius: 8px; font-size: 0.8em; }
  footer { text-align: center; color: var(--muted); font-size: 0.8em; margin-top: 32px; }
  .metric { display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 0.85em; }
  .metric-label { width: 76px; color: var(--muted); flex-shrink: 0; }
  .bar { flex: 1; height: 8px; background: #ececE4; border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; }
  .bar-fill.green { background: #4caf50; } .bar-fill.yellow { background: #e0b400; }
  .bar-fill.brown { background: #8d6e63; } .bar-fill.sat { background: #7e57c2; }
  .bar-fill.vigour { background: #26a69a; }
  .metric-value { width: 46px; text-align: right; font-variant-numeric: tabular-nums; color: var(--ink); flex-shrink: 0; }
  .causes { margin-top: 8px; }
  .cause { display: flex; gap: 10px; padding: 8px 0; border-top: 1px solid var(--line); font-size: 0.88em; line-height: 1.45; }
  .cause:first-child { border-top: none; }
  .cause-tag { flex-shrink: 0; font-size: 0.75em; font-weight: 700; text-transform: uppercase;
               letter-spacing: 0.03em; padding: 3px 8px; border-radius: 6px; height: fit-content;
               background: #eef0f3; color: var(--muted); white-space: nowrap; }
  .disclaimer { font-size: 0.82em; color: var(--muted); margin-top: 10px; font-style: italic; }
  .summary { font-size: 1.02em; line-height: 1.55; margin: 14px 0 0; padding: 12px 14px; background: #f7f6f1; border-radius: 10px; }
  .pill.sev-low { background: #558b2f; } .pill.sev-moderate { background: var(--amber); } .pill.sev-high { background: var(--red); }
  .trend { font-size: 0.92em; font-weight: 600; margin: 10px 0 0; }
  .trend-worsening { color: var(--red); } .trend-improving { color: var(--green); } .trend-stable { color: var(--muted); }
  .warn-box { background: #fff8e6; border: 1px solid #f0deab; border-radius: 10px; padding: 10px 14px; margin-top: 14px;
              font-size: 0.88em; line-height: 1.5; color: #6b5300; }
  .metric-label.wide { width: 190px; }
  .bar-fill.damage { background: #c62828; }
  .ctx { margin-top: 14px; } .ctx summary { color: var(--ink); }
  .ctx-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
  .ctx-grid label { font-size: 0.82em; color: var(--muted); display: flex; flex-direction: column; gap: 4px; }
  .ctx-grid input { padding: 7px 9px; border: 1px solid var(--line); border-radius: 8px; font-size: 0.95em; }
  @media (max-width: 520px) { .ctx-grid { grid-template-columns: 1fr; } .metric-label.wide { width: 120px; } }
</style></head>
<body>
<div class="wrap">
  <header>
    <div class="brand">M.A.X.I.K.</div>
    <h1>Tomato Health Check</h1>
    <p class="sub">Upload a photo of a tomato plant leaf. The system checks whether a plant is
    visible, whether it looks like a tomato, and flags possible visible symptoms — a screening
    aid, not a lab diagnosis.</p>
    <nav><a href="/">&larr; generic multi-plant pipeline</a></nav>
  </header>

  <div class="card">
    <form method=post enctype=multipart/form-data id="f">
      <div class="dropzone" id="dz" tabindex="0">
        <div>Drop a photo here, paste with Ctrl+V, or choose a file</div>
        <input type=file name=photo accept="image/*" required id="file-input">
      </div>
      <details class="ctx">
        <summary>Robot context (optional): plant point and sensor readings</summary>
        <div class="ctx-grid">
          <label>Point ID <input name="point" placeholder="e.g. row2-plant5" maxlength="64"></label>
          <label>Soil moisture (0-1) <input name="soil_moisture" type="number" step="0.01" min="0" max="1"></label>
          <label>Temperature (&deg;C) <input name="temperature" type="number" step="0.1"></label>
          <label>Humidity (0-1) <input name="humidity" type="number" step="0.01" min="0" max="1"></label>
        </div>
        <p class="disclaimer">With a point ID the result is saved to that plant's history, so the next
        photo of the same point shows whether it is getting better or worse. See
        <a href="/dashboard">the dashboard</a>.</p>
      </details>
      <button type=submit>Analyze photo</button>
    </form>
  </div>

  {% if result %}
  <div class="card">
    <div class="status-row">
      <p class="status">{{ result.statusLabel }}</p>
      <span class="pill {{ result.healthStatus|lower }}">{{ result.healthStatus.replace('_',' ') }}</span>
      {% if result.severity != "UNKNOWN" %}<span class="pill sev-{{ result.severity|lower }}">{{ result.severity }} severity</span>{% endif %}
    </div>

    {% if result.summary %}<p class="summary">{{ result.summary }}</p>{% endif %}

    {% if result.plantDetected %}
    <p class="meta">Plant: <b>{{ result.plantType }}</b>
      {% if result.plantConfidence is not none %}&middot; {{ '%.0f'|format(result.plantConfidence*100) }}% confidence{% endif %}
      {% if result.conditionConfidence is not none %}&middot; condition confidence {{ '%.0f'|format(result.conditionConfidence*100) }}%{% endif %}
      {% if result.pointId %}&middot; point <a href="/point/{{ result.pointId|urlencode }}">{{ result.pointId }}</a>{% endif %}
    </p>
    {% endif %}

    {% if result.trend and result.trend.direction != "unknown" %}
    <p class="trend trend-{{ result.trend.direction }}">
      {% if result.trend.direction == "worsening" %}&#9650; Worse than last check{% elif result.trend.direction == "improving" %}&#9660; Better than last check{% else %}&#9644; About the same as last check{% endif %}
      {% if result.trend.deltaAffected is not none %}({{ '%+.0f'|format(result.trend.deltaAffected*100) }} points of damaged area){% endif %}
    </p>
    {% endif %}

    {% if result.warnings %}
    <div class="warn-box">{% for w in result.warnings %}<div>&#9888; {{ w }}</div>{% endfor %}</div>
    {% endif %}

    {% if result.candidates and result.candidates|length > 1 %}
    <p class="section-label">What the model considered</p>
    {% for c in result.candidates %}
    <div class="metric"><span class="metric-label wide">{{ c.label }}</span>
      <div class="bar"><div class="bar-fill vigour" style="width:{{ '%.0f'|format(c.probability*100) }}%"></div></div>
      <span class="metric-value">{{ '%.0f'|format(c.probability*100) }}%</span></div>
    {% endfor %}
    {% endif %}

    <p class="message">{{ result.message }}</p>

    {% if result.observedFeatures %}
    <p class="section-label">Observed visual symptoms</p>
    <ul class="symptoms">{% for f in result.observedFeatures %}<li>{{ f }}</li>{% endfor %}</ul>
    {% endif %}

    {% if result.recommendations %}
    <p class="section-label">Recommendations</p>
    <ul class="recs">{% for r in result.recommendations %}<li>{{ r }}</li>{% endfor %}</ul>
    {% endif %}

    {% if not result.imageQuality.ok %}
    <div class="quality-warning">Image quality: {{ result.imageQuality.issues|join(', ') }}</div>
    {% endif %}

    {% if result.colorStats %}
    <p class="section-label">Pipeline measurements</p>
    {% if result.affectedFraction is not none %}
    <div class="metric"><span class="metric-label">Visible damage</span>
      <div class="bar"><div class="bar-fill damage" style="width:{{ '%.0f'|format(result.affectedFraction*100) }}%"></div></div>
      <span class="metric-value">{{ '%.0f'|format(result.affectedFraction*100) }}%</span></div>
    {% endif %}
    <div class="metric"><span class="metric-label">Green</span>
      <div class="bar"><div class="bar-fill green" style="width:{{ '%.0f'|format(result.colorStats.green_ratio*100) }}%"></div></div>
      <span class="metric-value">{{ '%.0f'|format(result.colorStats.green_ratio*100) }}%</span></div>
    <div class="metric"><span class="metric-label">Yellow</span>
      <div class="bar"><div class="bar-fill yellow" style="width:{{ '%.0f'|format(result.colorStats.yellow_ratio*100) }}%"></div></div>
      <span class="metric-value">{{ '%.0f'|format(result.colorStats.yellow_ratio*100) }}%</span></div>
    <div class="metric"><span class="metric-label">Brown</span>
      <div class="bar"><div class="bar-fill brown" style="width:{{ '%.0f'|format(result.colorStats.brown_ratio*100) }}%"></div></div>
      <span class="metric-value">{{ '%.0f'|format(result.colorStats.brown_ratio*100) }}%</span></div>
    <div class="metric"><span class="metric-label">Saturation</span>
      <div class="bar"><div class="bar-fill sat" style="width:{{ '%.0f'|format(result.colorStats.mean_saturation*100) }}%"></div></div>
      <span class="metric-value">{{ '%.0f'|format(result.colorStats.mean_saturation*100) }}%</span></div>
    <div class="metric"><span class="metric-label">Vigour (ExG)</span>
      <div class="bar"><div class="bar-fill vigour" style="width:{{ '%.0f'|format([result.colorStats.exg_mean,0]|max * 100) }}%"></div></div>
      <span class="metric-value">{{ '%.2f'|format(result.colorStats.exg_mean) }}</span></div>
    <div class="metric"><span class="metric-label">Plant in frame</span>
      <div class="bar"><div class="bar-fill green" style="width:{{ '%.0f'|format(result.imageQuality.plantAreaFraction*100) }}%"></div></div>
      <span class="metric-value">{{ '%.0f'|format(result.imageQuality.plantAreaFraction*100) }}%</span></div>
    <p class="disclaimer">Visible damage = share of the leaf area (inside its outline) that is not healthy
    green, estimated by colour only; severity LOW &lt;10%, MODERATE 10-30%, HIGH &gt;30%. It can miss
    diseases that don't change colour much (mosaic virus, mites, leaf curl). Green/yellow/brown = share
    of segmented plant pixels in that colour band. Saturation = colour intensity (low = pale).
    Vigour (ExG) = Excess-Green index. All of this is computed by maxik_cv.segment / lesion.py, not by the ML model.</p>
    {% if overlay_b64 %}
    <p class="section-label">What was measured (red = counted as damaged, outline = leaf area)</p>
    <img class="photo" src="data:image/jpeg;base64,{{ overlay_b64 }}">
    {% endif %}
    {% endif %}

    {% if result.possibleCauses %}
    <p class="section-label">Possible general causes (heuristic, not the ML diagnosis)</p>
    <div class="causes">
      {% for c in result.possibleCauses %}
      <div class="cause"><span class="cause-tag">{{ c.category }}{% if c.source == "sensor" %} &middot; sensor{% endif %}</span><span>{{ c.text }}</span></div>
      {% endfor %}
    </div>
    <p class="disclaimer">These are general possibilities based on the colour measurements above,
    not a diagnosis and not specific to any single condition above — several can apply at once,
    and some may not apply at all. Sources: Colorado State University Extension; general
    horticultural references.</p>
    {% endif %}

    {% if result.plantDetected and result.healthModelIsPlaceholder %}
    <div class="placeholder-note">Note for reviewers: no tomato-disease model is trained yet in
    this MVP, so real photos correctly land here by design — this demonstrates the pipeline
    wiring, not a finished diagnosis model.</div>
    {% endif %}
    {% if result.plantDetected and not result.healthModelIsPlaceholder %}
    <div class="placeholder-note">The disease model was trained on PlantVillage (public, lab-condition
    leaf photos) and scored {{ '%.1f'|format(model_test_acc*100) }}% on a held-out, de-duplicated split of that same
    dataset. That is not a field accuracy: it has not yet been validated on independently captured or
    robot-camera photos, where backgrounds, light and leaf angle differ.</div>
    {% endif %}

    <img class="photo" src="data:image/jpeg;base64,{{ orig_b64 }}">

    <details>
      <summary>Raw JSON result</summary>
      <pre>{{ result_json }}</pre>
    </details>
  </div>
  {% endif %}

  <footer>M.A.X.I.K. plant-health module &middot; local screening tool, not a substitute for expert diagnosis</footer>
</div>
<script>
  var dz = document.getElementById('dz'), input = document.getElementById('file-input');
  ['dragenter','dragover'].forEach(function(ev){ dz.addEventListener(ev, function(e){ e.preventDefault(); dz.classList.add('drag'); }); });
  ['dragleave','drop'].forEach(function(ev){ dz.addEventListener(ev, function(e){ e.preventDefault(); dz.classList.remove('drag'); }); });
  dz.addEventListener('drop', function(e){
    if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; document.getElementById('f').submit(); }
  });
  window.addEventListener('paste', function(e){
    var items = (e.clipboardData || window.clipboardData).items;
    if (!items) return;
    for (var i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') === 0) {
        var file = items[i].getAsFile();
        var dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
        document.getElementById('f').submit();
        break;
      }
    }
  });
</script>
</body></html>
"""


def _same(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


@app.before_request
def _ui_password():
    """Optional login for the browser pages (set MAXIK_UI_PASSWORD, any user name). The robot API
    has its own token and is exempt, so /api/health also works for load-balancer health checks."""
    password = os.environ.get("MAXIK_UI_PASSWORD")
    if not password or request.path.startswith("/api/"):
        return None
    auth = request.authorization
    if auth and _same(auth.password or "", password):
        return None
    return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="M.A.X.I.K."'})


_POINT_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _request_context() -> tuple[str | None, dict, str | None]:
    """(point_id, sensors, error). Point and sensors may come from the query string, the form or the
    X-Point-Id header, so a browser form, curl and the robot client all work the same way."""
    point = (request.values.get("point") or request.headers.get("X-Point-Id") or "").strip()
    if point and not _POINT_RE.match(point):
        return None, {}, "point ID may only contain letters, digits and _ . : - (max 64 characters)"
    return point or None, {k: request.values.get(k) for k in SENSOR_KEYS}, None


def _lesion_overlay_b64(img: np.ndarray, lesion) -> str:
    """The photo with what was measured drawn on it: leaf outline and the pixels counted as damaged."""
    if lesion is None or lesion.fraction is None:
        return ""
    vis = cv2.resize(img, (lesion.leaf_mask.shape[1], lesion.leaf_mask.shape[0]), interpolation=cv2.INTER_AREA)
    red = np.zeros_like(vis)
    red[..., 2] = 255
    m = lesion.affected_mask > 0
    vis[m] = cv2.addWeighted(vis, 0.45, red, 0.55, 0)[m]
    contours, _ = cv2.findContours(lesion.leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)
    return _to_b64(vis)


@app.route("/tomato", methods=["GET", "POST"])
def tomato():
    if request.method != "POST":
        return render_template_string(TOMATO_PAGE, result=None)

    point, sensors, error = _request_context()
    if error:
        return error, 400
    img = cv2.imdecode(np.frombuffer(request.files["photo"].read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return render_template_string(TOMATO_PAGE, result=None)

    analysis, _ = analyze(tomato_inspector, history, img, point, sensors)
    result = analysis.to_dict()
    return render_template_string(
        TOMATO_PAGE, result=result, orig_b64=_to_b64(img), overlay_b64=_lesion_overlay_b64(img, analysis.lesion),
        result_json=json.dumps(result, indent=2, ensure_ascii=False), model_test_acc=TOMATO_MODEL_TEST_ACC,
    )


@app.route("/dashboard")
def dashboard():
    return render_template_string(DASHBOARD_PAGE, **dashboard_context(history.points()))


@app.route("/point/<point_id>")
def point_page(point_id):
    if not _POINT_RE.match(point_id):
        return "unknown point", 404
    series = history.series(point_id)
    if not series:
        return "unknown point", 404
    return render_template_string(POINT_PAGE, **point_context(point_id, series))


@app.route("/thumb/<int:inspection_id>.jpg")
def thumb(inspection_id):
    data = history.thumb(inspection_id)
    if data is None:
        return "not found", 404
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


def _api_authorized() -> bool:
    expected = os.environ.get("MAXIK_API_TOKEN")
    if not expected:  # no token configured -> open (fine for localhost/LAN testing, not for a public URL)
        return True
    return _same(request.headers.get("Authorization", ""), f"Bearer {expected}")


@app.route("/api/health")
def api_health():
    return jsonify(status="ok", tokenRequired=bool(os.environ.get("MAXIK_API_TOKEN")))


@app.route("/api/tomato", methods=["POST"])
def api_tomato():
    """Robot endpoint: send one photo, get the analysis as JSON.
    Either multipart form field "photo", or the raw image bytes as the request body.
    Optional query parameters: point (plant/point ID - enables history and trend),
    soil_moisture (0-1), temperature (degC), humidity (0-1)."""
    if not _api_authorized():
        return jsonify(error="unauthorized"), 401
    point, sensors, error = _request_context()
    if error:
        return jsonify(error=error), 400
    raw = request.files["photo"].read() if "photo" in request.files else request.get_data()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
    if img is None:
        return jsonify(error="could not decode image"), 400
    analysis, inspection_id = analyze(tomato_inspector, history, img, point, sensors)
    return jsonify({**analysis.to_dict(), "inspectionId": inspection_id})


if __name__ == "__main__":
    # MAXIK_HOST=0.0.0.0 lets a robot on the same Wi-Fi/LAN reach the server; default stays local-only.
    app.run(host=os.environ.get("MAXIK_HOST", "127.0.0.1"), port=5055, debug=False)
