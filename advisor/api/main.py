"""
advisor/api/main.py — PART 12: FastAPI backend.

Serves the frozen forecasting stack + the whole advisor pipeline over HTTP, plus
the single-page dashboard. Heavy singletons (models, index) load lazily on first
use so start-up is instant.

    uvicorn advisor.api.main:app --reload --port 8000
    # open http://localhost:8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from advisor.config import CONFIG
from advisor.serving import get_forecast_service
from advisor.context_builder import ContextBuilder
from advisor.pipeline import get_pipeline

app = FastAPI(title="Delhi AQI Intelligence — Advisor API", version="1.0")

_STATE: dict = {}


def svc():
    if "svc" not in _STATE:
        _STATE["svc"] = get_forecast_service()
    return _STATE["svc"]


def pipeline():
    if "pipe" not in _STATE:
        _STATE["pipe"] = get_pipeline()
    return _STATE["pipe"]


def _resolve_t(time_index: int | None):
    s = svc()
    return s.resolve_time(time_index=time_index)


@app.get("/health")
def health():
    return {"status": "ok", "horizon": CONFIG.horizon,
            "llm_provider": CONFIG.llm_provider}


@app.get("/wards")
def wards():
    s = svc()
    nodes = s.d.nodes.sort_values("node_idx")
    return [{"ward_id": str(r.ward_id), "ward_name": str(r.ward_name),
             "lat": float(r.ward_lat), "lon": float(r.ward_lon),
             "node_idx": int(r.node_idx)} for r in nodes.itertuples()]


def _build_ward_geojson() -> dict:
    """Ward polygons (data/raw/gis/wards) joined to our ward_id by ward_name."""
    import json
    gj = json.loads((CONFIG.serving_dir.parent / "raw" / "gis" / "wards" /
                     "delhi_wards.geojson").read_text(encoding="utf-8"))
    nodes = svc().d.nodes
    name2 = {str(n.ward_name).strip().upper(): (str(n.ward_id), int(n.node_idx))
             for n in nodes.itertuples()}
    feats = []
    for f in gj["features"]:
        nm = (f["properties"].get("Ward_Name") or "").strip().upper()
        if nm not in name2:
            continue
        wid, nidx = name2[nm]
        f["properties"] = {"ward_id": wid, "node_idx": nidx, "ward_name": nm}
        feats.append(f)
    return {"type": "FeatureCollection", "features": feats}


@app.get("/wards.geojson")
def wards_geojson():
    if "geojson" not in _STATE:
        _STATE["geojson"] = _build_ward_geojson()
    return JSONResponse(_STATE["geojson"])


@app.get("/actions")
def actions():
    from advisor.config import ACTION_CATALOGUE, INTERVENTION_STAGES, ACTION_TARGET_SOURCE
    return [{"action": a, **ACTION_CATALOGUE[a],
             "stages": INTERVENTION_STAGES.get(a, []),
             "target_source": ACTION_TARGET_SOURCE.get(a, "")}
            for a in ACTION_CATALOGUE]


@app.get("/layers")
def layers(time_index: int | None = Query(None)):
    t = _resolve_t(time_index)
    key = ("layers", t)
    if key not in _STATE:
        _STATE[key] = svc().layers_all(t)
    return {"time_index": t, "time": str(svc().times[t]), "wards": _STATE[key]}


@app.get("/layers/source")
def layers_source(time_index: int | None = Query(None)):
    t = _resolve_t(time_index)
    key = ("source", t)
    if key not in _STATE:
        _STATE[key] = svc().source_all(t)
    return {"time_index": t, "source": _STATE[key]}


class SimRequest(BaseModel):
    ward_id: str
    sliders: dict[str, float] = {}      # {lever: 0..100}
    time_index: int | None = None


@app.post("/simulate")
def simulate(req: SimRequest):
    """Scenario Builder: compose the 7 policy sliders into feature multipliers,
    re-run the frozen GNN, and return before/after AQI, an approximate new source
    split, and a plain-language recommendation."""
    from advisor.config import SCENARIO_SLIDERS, SLIDER_META
    from advisor.simulation.counterfactual import CounterfactualSimulator
    from advisor.context_builder import ContextBuilder
    s = svc()
    try:
        node = s.ward_to_node(req.ward_id)
    except ValueError:
        raise HTTPException(404, f"ward {req.ward_id} not found")
    t = _resolve_t(req.time_index)

    # compose feature multipliers from the sliders
    merged: dict[str, float] = {}
    active = []
    for lever, pct in req.sliders.items():
        if pct <= 0 or lever not in SCENARIO_SLIDERS:
            continue
        active.append((lever, pct))
        inten = pct / 100.0
        for feat, mult in SCENARIO_SLIDERS[lever].items():
            eff = 1.0 - inten * (1.0 - mult)
            merged[feat] = merged.get(feat, 1.0) * eff

    sim = CounterfactualSimulator(s)
    cf = sim.simulate_raw(node, t, merged).as_dict() if merged else None
    wc = ContextBuilder(s).build(node_idx=node, time_index=t, include_explanations=False)
    before_sources = wc.dominant_sources

    # 9.3 — TRUE post-intervention source split: re-run the attribution heads on
    # the modified features (not an approximate share-shrink).
    after_sources = s.attribution_after(node, t, merged) if merged else before_sources

    aqi_before = wc.predicted_aqi
    aqi_after = cf["aqi_after"] if cf else aqi_before
    top_lever = max(active, key=lambda lp: lp[1])[0] if active else None
    rec = (f"Prioritise '{SLIDER_META[top_lever]['label']}' — it targets "
           f"{SLIDER_META[top_lever]['source']}, the leading modelled contributor here."
           if top_lever else "Move the sliders to simulate an intervention.")
    return {"ward_id": req.ward_id, "aqi_before": aqi_before, "aqi_after": aqi_after,
            "delta": round(aqi_after - aqi_before, 1),
            "improvement_pct": round(100 * (aqi_before - aqi_after) / aqi_before, 1) if aqi_before else 0,
            "after_band": cf["after_band"] if cf else None,        # 9.4
            "before_sources": before_sources, "after_sources": after_sources,
            "applied_features": cf["applied_features"] if cf else {},
            "active_levers": [l for l, _ in active], "recommendation": rec}


@app.get("/cost_effectiveness")
def cost_effectiveness(ward_id: str, time_index: int | None = None):
    """9.5 — AQI drop per unit cost for each action (a decision-ranking aid)."""
    from advisor.simulation.counterfactual import CounterfactualSimulator
    from advisor.config import ACTION_CATALOGUE
    s = svc()
    try:
        node = s.ward_to_node(ward_id)
    except ValueError:
        raise HTTPException(404, f"ward {ward_id} not found")
    t = _resolve_t(time_index)
    sim = CounterfactualSimulator(s)
    out = []
    for aid, meta in ACTION_CATALOGUE.items():
        r = sim.simulate_action(node, t, aid)
        drop = max(-r.delta, 0.0)
        cost = float(meta.get("cost", 0.5))
        out.append({"action": aid, "label": meta.get("label", aid),
                    "aqi_drop": round(drop, 1), "cost": cost,
                    "time_to_effect_h": meta.get("time_to_effect_h"),
                    "effectiveness": round(drop / (cost + 0.05), 2)})
    out.sort(key=lambda d: -d["effectiveness"])
    return {"ward_id": ward_id, "actions": out}


@app.get("/explain/global")
def explain_global(n_queries: int = 200):
    """3.2 — global feature importance (mean |SHAP| across many ward-hours)."""
    if "shap_global" not in _STATE:
        from explain_gnn import shap_global
        g = shap_global(svc().ctx, n_queries=n_queries)
        _STATE["shap_global"] = [{"feature": r.feature,
                                  "mean_abs_shap_aqi": round(float(r.mean_abs_shap_aqi), 3)}
                                 for r in g.head(15).itertuples()]
    return {"global_importance": _STATE["shap_global"]}


@app.get("/explain/stability")
def explain_stability(ward_id: str, time_index: int | None = None, seeds: int = 3):
    """3.5 — GNNExplainer seed-stability (is the explanation trustworthy?)."""
    from explain_gnn import gnn_stability
    s = svc()
    try:
        node = s.ward_to_node(ward_id)
    except ValueError:
        raise HTTPException(404, f"ward {ward_id} not found")
    return gnn_stability(s.ctx, node, _resolve_t(time_index), seeds=seeds)


@app.get("/kg")
def knowledge_graph_view():
    """The knowledge graph as display-ready nodes + links (6.3)."""
    from advisor.kg.knowledge_graph import get_knowledge_graph
    kg = get_knowledge_graph().g
    nodes, links = [], []
    for n, d in kg.nodes(data=True):
        nt = d.get("ntype", "other")
        label = d.get("title", n) if nt == "policy" else str(n)
        nodes.append({"id": str(n), "type": nt, "label": label[:32]})
    for u, v, d in kg.edges(data=True):
        links.append({"source": str(u), "target": str(v), "relation": d.get("relation", "")})
    return {"nodes": nodes, "links": links,
            "node_types": sorted({d.get("ntype", "other") for _, d in kg.nodes(data=True)})}


@app.get("/reference")
def reference():
    from advisor.config import (AQI_BANDS, HEALTH_ADVICE, WHO_LIMITS, SOURCE_COLORS,
                                GRAP_STAGES, SLIDER_META, MAP_LAYERS)
    return {"bands": [{"lo": b[0], "hi": b[1], "name": b[2], "color": b[3]} for b in AQI_BANDS],
            "health_advice": HEALTH_ADVICE, "who_limits": WHO_LIMITS,
            "source_colors": SOURCE_COLORS,
            "grap_stages": {k: {"lo": v[0], "hi": v[1], "band": v[2]} for k, v in GRAP_STAGES.items()},
            "sliders": SLIDER_META, "map_layers": MAP_LAYERS,
            "n_times": int(len(svc().times)), "latest_index": svc().latest_time_index()}


@app.get("/realtime")
def realtime_status():
    """Live ingestion state (§1.7): whether forecasts are served from live data,
    when it was last refreshed, and the basis hour the dashboard is showing."""
    import json
    s = svc()
    t = s.latest_time_index()
    status_path = CONFIG.serving_dir.parent / "realtime" / "status.json"
    ingest = json.loads(status_path.read_text()) if status_path.exists() else None
    return {"live": bool(CONFIG.realtime), "basis_time": str(s.times[t]),
            "basis_index": t, "data_max_time": str(s.times.max()), "ingest": ingest}


@app.get("/map")
def map_layer(time_index: int | None = Query(None)):
    t = _resolve_t(time_index)
    return {"time_index": t, "time": str(svc().times[t]), "wards": svc().forecast_all(t)}


@app.get("/forecast")
def forecast(ward_id: str, time_index: int | None = None, horizon: int = 24):
    s = svc()
    try:
        node = s.ward_to_node(ward_id)
    except ValueError:
        raise HTTPException(404, f"ward {ward_id} not found")
    t = _resolve_t(time_index)
    fc = s.forecast(node, t, horizon=horizon)
    return {**fc.as_dict(), "ward_id": ward_id, "time": str(fc.time),
            "available_horizons": s.available_horizons(),
            "history": s.history(node, t, hours=48)}


@app.get("/heatmap")
def heatmap(ward_id: str):
    """11.1 — day-of-week × hour AQI heatmap for a ward."""
    s = svc()
    try:
        node = s.ward_to_node(ward_id)
    except ValueError:
        raise HTTPException(404, f"ward {ward_id} not found")
    if ("heatmap", node) not in _STATE:
        _STATE[("heatmap", node)] = s.aqi_heatmap(node)
    return _STATE[("heatmap", node)]


@app.get("/windrose")
def windrose(ward_id: str, time_index: int | None = None, hours: int = 48):
    s = svc()
    try:
        node = s.ward_to_node(ward_id)
    except ValueError:
        raise HTTPException(404, f"ward {ward_id} not found")
    return s.wind_rose(node, _resolve_t(time_index), hours=hours)


@app.get("/context")
def context(ward_id: str, time_index: int | None = None, explain: bool = False):
    try:
        wc = ContextBuilder(svc()).build(ward_id=ward_id, time_index=time_index,
                                         include_explanations=explain)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return wc.to_dict()


@app.get("/advise")
def advise(ward_id: str, time_index: int | None = None, explain: bool = False,
           weights: str | None = None):
    import json as _json
    w = None
    if weights:                            # 10.2 — user-tunable objective weights (JSON)
        try:
            w = {k: float(v) for k, v in _json.loads(weights).items()}
        except Exception:
            raise HTTPException(400, "weights must be a JSON object of {objective: number}")
    try:
        return pipeline().advise(ward_id=ward_id, time_index=time_index,
                                 include_explanations=explain, ranking_weights=w)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/portfolio")
def portfolio(ward_id: str, budget: float = 1.5, time_index: int | None = None):
    """10.4 — best COMBINATION of actions under a cost budget (greedy by AQI-drop
    per cost), with the combined counterfactual of the chosen set."""
    from advisor.simulation.counterfactual import CounterfactualSimulator
    from advisor.config import ACTION_CATALOGUE, INTERVENTION_STAGES, grap_stage
    s = svc()
    try:
        node = s.ward_to_node(ward_id)
    except ValueError:
        raise HTTPException(404, f"ward {ward_id} not found")
    t = _resolve_t(time_index)
    sim = CounterfactualSimulator(s)
    stage = grap_stage(s.forecast(node, t).p50)
    # candidate actions applicable at this stage, scored by drop-per-cost
    cands = []
    for aid, meta in ACTION_CATALOGUE.items():
        if stage and INTERVENTION_STAGES.get(aid) and stage not in INTERVENTION_STAGES[aid]:
            continue
        drop = max(-sim.simulate_action(node, t, aid).delta, 0.0)
        cost = float(meta.get("cost", 0.5))
        cands.append({"action": aid, "label": meta.get("label", aid), "cost": cost,
                      "solo_drop": round(drop, 1), "eff": drop / (cost + 0.05)})
    cands.sort(key=lambda d: -d["eff"])
    chosen, spent = [], 0.0
    for c in cands:
        if spent + c["cost"] <= budget:
            chosen.append(c); spent += c["cost"]
    combined = sim.simulate_actions(node, t, [c["action"] for c in chosen]).as_dict() if chosen else None
    return {"ward_id": ward_id, "budget": budget, "spent_cost": round(spent, 2),
            "grap_stage": stage, "chosen": chosen,
            "combined_aqi_before": combined["aqi_before"] if combined else None,
            "combined_aqi_after": combined["aqi_after"] if combined else None,
            "combined_improvement_pct": combined["improvement_pct"] if combined else 0}


class CFRequest(BaseModel):
    ward_id: str
    actions: list[str] = []
    feature_multipliers: dict[str, float] | None = None
    intensity: float = 1.0
    spillover: bool = False               # 9.2 — also affect neighbouring wards
    time_index: int | None = None


@app.post("/counterfactual")
def counterfactual(req: CFRequest):
    from advisor.simulation.counterfactual import CounterfactualSimulator, _scaled_multipliers
    s = svc()
    try:
        node = s.ward_to_node(req.ward_id)
    except ValueError:
        raise HTTPException(404, f"ward {req.ward_id} not found")
    t = _resolve_t(req.time_index)
    sim = CounterfactualSimulator(s)
    mults = req.feature_multipliers or _scaled_multipliers(req.actions, req.intensity)
    return sim.simulate_raw(node, t, mults, spillover=req.spillover).as_dict()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    html = (Path(__file__).resolve().parent.parent / "dashboard" / "index.html")
    if not html.exists():
        return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
    return HTMLResponse(html.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
