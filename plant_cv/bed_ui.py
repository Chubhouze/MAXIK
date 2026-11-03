"""Browser pages for the top-down bed view: /bed (analyse one photo), /beds/<id> (a robot pass along a bed:
crops, weeds and damage by position). Data from maxik_cv.tomato.bed and TomatoHistory.bed_frames."""
import cv2
import numpy as np

from dashboard_ui import CSS, _when

EXTRA_CSS = """
  .dropzone { border: 2px dashed var(--line); border-radius: var(--radius); padding: 22px; text-align: center; color: var(--muted); }
  button { background: var(--ink); color: #fff; border: none; border-radius: 999px; padding: 9px 20px; font-size: .95em; cursor: pointer; margin-top: 12px; }
  .summary { font-size: 1.02em; line-height: 1.55; padding: 12px 14px; background: #f7f6f1; border-radius: 10px; }
  .warn { background: #fff8e6; border: 1px solid #f0deab; border-radius: 10px; padding: 9px 13px; margin-top: 12px; font-size: .87em; color: #6b5300; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
  .grid2 label { font-size: .82em; color: var(--muted); display: flex; flex-direction: column; gap: 4px; }
  .grid2 input { padding: 7px 9px; border: 1px solid var(--line); border-radius: 8px; }
  .photo { max-width: 100%; border-radius: 10px; margin-top: 14px; display: block; }
  .legend span { display: inline-block; margin-right: 14px; font-size: .85em; color: var(--muted); }
  .sw { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
  .stat { display: inline-block; margin-right: 22px; } .stat b { font-size: 1.4em; display: block; }
  @media (max-width: 520px) { .grid2 { grid-template-columns: 1fr; } }
"""

BED_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>M.A.X.I.K. — Bed View</title><style>{{ css|safe }}</style></head><body><div class="wrap">
<div class="brand">M.A.X.I.K.</div><h1>Bed view (camera looking down)</h1>
<p class="sub">Upload a top-down photo of a bed. Every plant is found and marked as crop or weed; each crop plant's
visible damage is measured. Disease names need a close-up photo: use <a href="/tomato">single plant</a> for that.</p>
<nav><a href="/tomato">single plant</a><a href="/dashboard">inspections</a><a href="/beds">bed passes</a></nav>
{% if not detector_ready %}<div class="warn">No bed detector model found (models_bed/bed_detector.onnx).</div>{% endif %}
<div class="card"><form method=post enctype=multipart/form-data>
  <div class="dropzone">Top-down photo of a bed <input type=file name=photo accept="image/*" required></div>
  <div class="grid2"><label>Bed ID (optional, saves the frame) <input name="bed" maxlength="64" placeholder="bed-1"></label>
  <label>Position along the bed, cm (optional) <input name="position" type="number" step="any"></label></div>
  <button type=submit>Analyse bed photo</button></form></div>
{% if report %}
<div class="card">
  <div><span class="stat"><b>{{ report.crops }}</b>crop plants</span><span class="stat"><b>{{ report.damagedCrops }}</b>with visible damage</span>
  <span class="stat"><b>{{ report.weeds }}</b>weeds</span><span class="stat"><b>{{ '%.0f'|format(report.weedCover*100) }}%</b>weed cover</span></div>
  <p class="summary">{{ report.summary }}</p>
  <div class="legend"><span><i class="sw" style="background:#2e7d32"></i>crop, little damage</span><span><i class="sw" style="background:#ef8c00"></i>crop, 10-30% damaged</span>
  <span><i class="sw" style="background:#c62828"></i>crop, &gt;30% damaged</span><span><i class="sw" style="background:#8e24aa"></i>weed</span></div>
  <img class="photo" src="data:image/jpeg;base64,{{ overlay_b64 }}">
  {% for w in report.warnings %}<div class="warn">&#9888; {{ w }}</div>{% endfor %}
  {% if report.plants %}
  <div class="scroll"><table><tr><th>#</th><th>Type</th><th>Visible damage</th><th>Severity (visual)</th><th>Detector confidence</th></tr>
  {% for p in report.plants %}<tr><td>{{ p.index }}</td><td>{{ p.kind }}</td>
  <td class="num">{% if p.affectedFraction is not none %}{{ '%.0f'|format(p.affectedFraction*100) }}%{% else %}&ndash;{% endif %}</td>
  <td>{% if p.kind == 'crop' and p.severity != 'UNKNOWN' %}<span class="pill sev-{{ p.severity|lower }}">{{ p.severity }}</span>{% endif %}</td>
  <td class="num">{{ '%.0f'|format(p.score*100) }}%</td></tr>{% endfor %}</table></div>
  {% endif %}
  {% if report.possibleCauses %}<p class="sub" style="margin-top:14px"><b>Possible general causes</b> (colour of the crop plants{% if report.possibleCauses|selectattr('source','equalto','sensor')|list %} + sensors{% endif %}):</p>
  <ul>{% for c in report.possibleCauses %}<li><b>{{ c.category }}</b>: {{ c.text }}</li>{% endfor %}</ul>{% endif %}
</div>
{% endif %}
<footer>Screening aid, not a laboratory diagnosis. Bed detector trained on CropAndWeed (non-commercial licence).</footer>
</div></body></html>"""

BEDS_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>M.A.X.I.K. — Bed Passes</title><style>{{ css|safe }}</style></head><body><div class="wrap">
<div class="brand">M.A.X.I.K.</div><h1>{% if bed_id %}Bed {{ bed_id }}{% else %}Bed passes{% endif %}</h1>
<nav><a href="/bed">analyse a bed photo</a><a href="/beds">all beds</a><a href="/dashboard">inspections</a></nav>
{% if not bed_id %}
<div class="card scroll">{% if beds %}<table><tr><th>Bed</th><th>Frames</th><th>Last frame</th><th>Crops</th><th>Damaged</th><th>Weeds</th></tr>
{% for b in beds %}<tr><td><a href="/beds/{{ b.bed_id|urlencode }}"><b>{{ b.bed_id }}</b></a></td><td class="num">{{ b.frames }}</td>
<td class="num">{{ b.when }}</td><td class="num">{{ b.crops }}</td><td class="num">{{ b.damaged }}</td><td class="num">{{ b.weeds }}</td></tr>{% endfor %}</table>
{% else %}<div class="empty">No bed frames yet. The robot sends them with <code>python3 scripts/patrol_bed.py --url ... --bed bed-1</code>,
or enter a Bed ID on the <a href="/bed">bed page</a>.</div>{% endif %}</div>
{% else %}
<div class="card"><b>Along the bed</b><p class="sub">Each frame the robot sent, by position (or order, if no position was sent):
crop plants (green), those with visible damage (red) and weeds (purple).</p>
<div class="scroll"><svg width="{{ chart.w }}" height="{{ chart.h }}" viewBox="0 0 {{ chart.w }} {{ chart.h }}">
<line x1="40" x2="{{ chart.w - 10 }}" y1="{{ chart.base }}" y2="{{ chart.base }}" stroke="#d9d9d0"/>
{% for g in chart.ticks %}<text x="34" y="{{ g.y + 4 }}" text-anchor="end">{{ g.label }}</text><line x1="40" x2="{{ chart.w - 10 }}" y1="{{ g.y }}" y2="{{ g.y }}" stroke="#efefe8"/>{% endfor %}
{% for f in chart.frames %}
<rect x="{{ f.x }}" y="{{ f.crop_y }}" width="{{ chart.bw }}" height="{{ f.crop_h }}" fill="#2e7d32"><title>{{ f.tip }}</title></rect>
<rect x="{{ f.x }}" y="{{ f.dmg_y }}" width="{{ chart.bw }}" height="{{ f.dmg_h }}" fill="#c62828"><title>{{ f.tip }}</title></rect>
<rect x="{{ f.x + chart.bw + 1 }}" y="{{ f.weed_y }}" width="{{ chart.bw }}" height="{{ f.weed_h }}" fill="#8e24aa"><title>{{ f.tip }}</title></rect>
<text x="{{ f.x + chart.bw }}" y="{{ chart.h - 6 }}" text-anchor="middle">{{ f.label }}</text>
{% endfor %}</svg></div></div>
<div class="card scroll"><table><tr><th></th><th>Position</th><th>When</th><th>Crops</th><th>Damaged</th><th>Mean damage</th><th>Weeds</th></tr>
{% for f in frames|reverse %}<tr><td><img class="thumb" src="/bedthumb/{{ f.id }}.jpg" alt=""></td><td class="num">{{ f.position if f.position is not none else '–' }}</td>
<td class="num">{{ f.when }}</td><td class="num">{{ f.crops }}</td><td class="num">{{ f.damaged }}</td>
<td class="num">{% if f.mean_damage is not none %}{{ '%.0f'|format(f.mean_damage*100) }}%{% else %}–{% endif %}</td><td class="num">{{ f.weeds }}</td></tr>{% endfor %}</table></div>
{% endif %}
<footer>Screening aid, not a laboratory diagnosis.</footer></div></body></html>"""


def css() -> str:
    return CSS + EXTRA_CSS


def overlay(bgr: np.ndarray, report: dict) -> np.ndarray:
    s = min(1.0, 1100 / max(bgr.shape[:2]))
    vis = cv2.resize(bgr, (round(bgr.shape[1] * s), round(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
    thick = max(2, round(max(vis.shape[:2]) / 400))
    for p in report["plants"]:
        x0, y0, x1, y1 = (round(v * s) for v in p["box"])
        if p["kind"] == "weed":
            colour = (170, 36, 142)
        else:
            a = p["affectedFraction"] or 0
            colour = (40, 40, 198) if a >= 0.30 else (0, 140, 239) if a >= 0.10 else (50, 125, 46)
        cv2.rectangle(vis, (x0, y0), (x1, y1), colour, thick)
        if p["kind"] == "crop":
            label = str(p["index"])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x0, y0), (x0 + tw + 6, y0 + th + 6), colour, -1)
            cv2.putText(vis, label, (x0 + 3, y0 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def beds_context(beds: list[dict]) -> dict:
    return {"css": css(), "beds": [{**b, "when": _when(b["last_ts"])} for b in beds], "bed_id": None}


def bed_context(bed_id: str, frames: list[dict]) -> dict:
    w, h, base, bw = 720, 220, 190, 8
    top = max([max(f["crops"] or 0, f["weeds"] or 0) for f in frames] + [1])
    n = max(1, len(frames))
    step = (w - 60) / n
    out = []
    for i, f in enumerate(sorted(frames, key=lambda f: (f["position"] is None, f["position"] or 0, f["id"]))):
        k = (base - 20) / top
        x = 44 + i * step
        crop_h, dmg_h, weed_h = (f["crops"] or 0) * k, (f["damaged"] or 0) * k, (f["weeds"] or 0) * k
        label = f"{f['position']:g}" if f["position"] is not None else str(i + 1)
        out.append({"x": round(x, 1), "crop_y": round(base - crop_h, 1), "crop_h": round(crop_h, 1),
                    "dmg_y": round(base - dmg_h, 1), "dmg_h": round(dmg_h, 1), "weed_y": round(base - weed_h, 1),
                    "weed_h": round(weed_h, 1), "label": label,
                    "tip": f"{label}: {f['crops']} crops ({f['damaged']} damaged), {f['weeds']} weeds"})
    ticks = [{"y": round(base - (base - 20) * t / top, 1), "label": str(t)} for t in sorted({0, top // 2, top})]
    return {"css": css(), "bed_id": bed_id, "frames": [{**f, "when": _when(f["ts"])} for f in frames],
            "chart": {"w": w, "h": h, "base": base, "bw": bw, "frames": out, "ticks": ticks}}
