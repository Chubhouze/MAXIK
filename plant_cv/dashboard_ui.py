"""Browser pages for the robot's inspection history: /dashboard (every point, latest state, damage
sparkline) and /point/<id> (damage over time with the severity bands, every inspection listed).
Pure presentation - the data comes from maxik_cv.tomato.history.TomatoHistory.
"""
from datetime import datetime

from maxik_cv.tomato.history import STABLE_BAND

CSS = """
  :root { --green:#2e7d32; --red:#c62828; --amber:#b8620a; --gray:#6b7280; --ink:#1f2430; --muted:#6b7280;
          --line:#e6e6e0; --card:#fff; --bg:#f4f3ee; --radius:14px; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--ink); margin: 0; padding: 32px 16px 64px; }
  .wrap { max-width: 860px; margin: 0 auto; }
  .brand { font-size: .8em; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
  h1 { font-size: 1.6em; margin: 4px 0 8px; }
  .sub { color: var(--muted); font-size: .95em; line-height: 1.5; margin: 0; }
  nav { margin-top: 10px; font-size: .9em; } nav a { color: var(--muted); margin-right: 14px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius); padding: 20px; margin-top: 20px; }
  .pill { display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .78em; font-weight: 700; color: #fff; white-space: nowrap; }
  .healthy { background: var(--green); } .unhealthy { background: var(--red); } .uncertain { background: var(--amber); }
  .sev-low { background: #558b2f; } .sev-moderate { background: var(--amber); } .sev-high { background: var(--red); }
  .sev-unknown { background: var(--gray); }
  table { width: 100%; border-collapse: collapse; font-size: .92em; }
  th { text-align: left; font-size: .75em; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); padding: 6px 8px; }
  td { padding: 10px 8px; border-top: 1px solid var(--line); vertical-align: middle; }
  td.num { font-variant-numeric: tabular-nums; }
  img.thumb { width: 64px; height: 64px; object-fit: cover; border-radius: 8px; display: block; }
  .scroll { overflow-x: auto; }
  .up { color: var(--red); font-weight: 700; } .down { color: var(--green); font-weight: 700; } .flat { color: var(--muted); }
  .empty { color: var(--muted); padding: 24px 0; text-align: center; line-height: 1.6; }
  code { background: #f1f0ea; padding: 2px 6px; border-radius: 5px; font-size: .9em; }
  svg text { font-family: inherit; fill: var(--muted); font-size: 11px; }
  footer { text-align: center; color: var(--muted); font-size: .8em; margin-top: 32px; }
"""

DASHBOARD_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>M.A.X.I.K. — Plant Inspections</title><style>{{ css|safe }}</style></head><body><div class="wrap">
<div class="brand">M.A.X.I.K.</div><h1>Plant inspections</h1>
<p class="sub">Latest state of every plant point the robot has reported, newest visit first. Damage is the
share of visible leaf area that is not healthy green; the line shows how it changed between visits.</p>
<nav><a href="/tomato">&larr; analyze a photo</a></nav>
<div class="card scroll">
{% if points %}
<table><tr><th></th><th>Point</th><th>Last visit</th><th>Assessment</th><th>Damage</th><th>Over time</th><th>Visits</th></tr>
{% for p in points %}
<tr>
  <td>{% if p.latest.id %}<img class="thumb" src="/thumb/{{ p.latest.id }}.jpg" alt=""> {% endif %}</td>
  <td><a href="/point/{{ p.point_id|urlencode }}"><b>{{ p.point_id }}</b></a></td>
  <td class="num">{{ p.when }}</td>
  <td>{{ p.latest.status_label }}<br>
      <span class="pill {{ p.health_class }}">{{ p.health_text }}</span>
      {% if p.latest.severity != "UNKNOWN" %}<span class="pill sev-{{ p.latest.severity|lower }}">{{ p.latest.severity }}</span>{% endif %}</td>
  <td class="num">{% if p.latest.affected_fraction is not none %}{{ '%.0f'|format(p.latest.affected_fraction*100) }}%{% else %}&ndash;{% endif %}
      <span class="{{ p.trend_class }}">{{ p.trend_arrow }}</span></td>
  <td>{% if p.spark %}<svg width="120" height="30" viewBox="0 0 120 30"><polyline points="{{ p.spark }}" fill="none" stroke="#c62828" stroke-width="2"/></svg>{% endif %}</td>
  <td class="num">{{ p.count }}</td>
</tr>
{% endfor %}</table>
{% else %}
<div class="empty">No inspections yet.<br>Send a photo with a point ID, for example
<code>python3 scripts/send_to_server.py --url ... --point row1-plant3</code>,
or fill in "Point ID" under <a href="/tomato">Robot context</a> in the web form.</div>
{% endif %}
</div>
<footer>Screening aid, not a laboratory diagnosis.</footer></div></body></html>"""

POINT_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ point_id }} — M.A.X.I.K.</title><style>{{ css|safe }}</style></head><body><div class="wrap">
<div class="brand">M.A.X.I.K. &middot; plant point</div><h1>{{ point_id }}</h1>
<nav><a href="/dashboard">&larr; all points</a><a href="/tomato">analyze a photo</a></nav>

<div class="card">
<b>Visible damage over time</b>
<p class="sub">Share of leaf area not healthy green at each visit (visits are evenly spaced, not to time scale).
Dashed lines: 10% (above = moderate) and 30% (above = high).</p>
{% if chart.dots %}
<div class="scroll"><svg width="{{ chart.w }}" height="{{ chart.h }}" viewBox="0 0 {{ chart.w }} {{ chart.h }}">
  {% for g in chart.grid %}<line x1="{{ chart.left }}" x2="{{ chart.w - 10 }}" y1="{{ g.y }}" y2="{{ g.y }}" stroke="{{ '#d9d9d0' if g.dashed else '#ececE4' }}" {% if g.dashed %}stroke-dasharray="4 4"{% endif %}/>
  <text x="{{ chart.left - 6 }}" y="{{ g.y + 4 }}" text-anchor="end">{{ g.label }}</text>{% endfor %}
  {% if chart.line %}<polyline points="{{ chart.line }}" fill="none" stroke="#c62828" stroke-width="2"/>{% endif %}
  {% for d in chart.dots %}<circle cx="{{ d.x }}" cy="{{ d.y }}" r="4" fill="#c62828"><title>{{ d.tip }}</title></circle>
  <text x="{{ d.x }}" y="{{ chart.h - 8 }}" text-anchor="middle">{{ d.label }}</text>{% endfor %}
</svg></div>
{% else %}<div class="empty">No measurable damage values yet.</div>{% endif %}
</div>

<div class="card scroll">
<b>Inspections</b>
<table><tr><th></th><th>When</th><th>Assessment</th><th>Damage</th><th>Sensors</th></tr>
{% for r in rows %}
<tr><td><img class="thumb" src="/thumb/{{ r.id }}.jpg" alt=""></td>
<td class="num">{{ r.when }}</td>
<td>{{ r.status_label }}<br><span class="pill {{ r.health_class }}">{{ r.health_text }}</span>
    {% if r.severity != "UNKNOWN" %}<span class="pill sev-{{ r.severity|lower }}">{{ r.severity }}</span>{% endif %}</td>
<td class="num">{% if r.affected_fraction is not none %}{{ '%.0f'|format(r.affected_fraction*100) }}%{% else %}&ndash;{% endif %}</td>
<td class="num">{{ r.sensor_text }}</td></tr>
{% endfor %}</table>
</div>
<footer>Screening aid, not a laboratory diagnosis.</footer></div></body></html>"""


def _when(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return ts


def _health(condition: str) -> tuple[str, str]:
    if condition == "TOMATO_HEALTHY":
        return "healthy", "HEALTHY"
    if condition == "TOMATO_UNCERTAIN":
        return "uncertain", "UNCERTAIN"
    return "unhealthy", "UNHEALTHY"


def sparkline(values: list, w: int = 120, h: int = 30, top: float = 0.5) -> str:
    pts = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(pts) < 2:
        return ""
    top = max(top, max(v for _, v in pts))
    n = len(values) - 1
    return " ".join(f"{2 + (w - 4) * i / n:.1f},{h - 3 - (h - 6) * v / top:.1f}" for i, v in pts)


def _trend(damage: list) -> tuple[str, str]:
    seen = [v for v in damage if v is not None]
    if len(seen) < 2:
        return "", "flat"
    d = seen[-1] - seen[-2]
    if d >= STABLE_BAND:
        return "▲", "up"
    if d <= -STABLE_BAND:
        return "▼", "down"
    return "▬", "flat"


def dashboard_context(points: list[dict]) -> dict:
    out = []
    for p in points:
        cls, text = _health(p["latest"]["condition"])
        arrow, tcls = _trend(p["damage"])
        out.append({**p, "when": _when(p["latest"]["ts"]), "health_class": cls, "health_text": text,
                    "spark": sparkline(p["damage"]), "trend_arrow": arrow, "trend_class": tcls})
    return {"points": out, "css": CSS}


def point_context(point_id: str, series: list[dict]) -> dict:
    w, h, left, top_pad, bottom = 640, 230, 44, 12, 28
    vals = [s["affected_fraction"] for s in series if s["affected_fraction"] is not None]
    ymax = max(0.4, (max(vals) + 0.05) if vals else 0.4)

    def y(v):
        return top_pad + (h - top_pad - bottom) * (1 - v / ymax)

    grid = [{"y": y(0), "label": "0%", "dashed": False}]
    for level in (0.10, 0.30):
        if level <= ymax:
            grid.append({"y": y(level), "label": f"{level:.0%}", "dashed": True})
    n = max(1, len(series) - 1)
    days = [_when(s["ts"])[5:10] for s in series]
    same_day = len(set(days)) < len(days)  # several visits a day: label with the time instead
    dots, line = [], []
    for i, s in enumerate(series):
        v = s["affected_fraction"]
        if v is None:
            continue
        x = left + (w - left - 16) * (i / n if len(series) > 1 else 0.5)
        dots.append({"x": round(x, 1), "y": round(y(v), 1),
                     "label": _when(s["ts"])[11:16] if same_day else days[i],
                     "tip": f"{_when(s['ts'])}: {v:.0%} damaged, {s['status_label']}"})
        line.append(f"{x:.1f},{y(v):.1f}")
    rows = []
    for s in reversed(series):
        cls, text = _health(s["condition"])
        sensors = s["sensors"]
        bits = [f"soil {sensors['soil_moisture']:.2f}" if "soil_moisture" in sensors else "",
                f"{sensors['temperature']:.0f} °C" if "temperature" in sensors else "",
                f"RH {sensors['humidity']:.0%}" if "humidity" in sensors else ""]
        rows.append({**s, "when": _when(s["ts"]), "health_class": cls, "health_text": text,
                     "sensor_text": ", ".join(b for b in bits if b) or "–"})
    return {"point_id": point_id, "css": CSS, "rows": rows,
            "chart": {"w": w, "h": h, "left": left, "grid": grid, "dots": dots,
                      "line": " ".join(line) if len(line) > 1 else ""}}
