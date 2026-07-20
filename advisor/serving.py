"""
advisor/serving.py — the reuse layer over the frozen forecasting stack.

`ForecastService` is the ONLY place the advisor touches the trained models. It
composes, without duplicating or retraining anything:

  * `explain_gnn.ExplainContext`  -> the Graph-Transformer checkpoint (predict +
    SHAP + GNNExplainer) and the processed data tensors.
  * `models.attribution.SourceAttributor` -> the LightGBM source heads.
  * `advisor.feature_space` -> raw display values + raw<->scaled transforms.

Everything downstream (Context Builder, Counterfactual, Dashboard) calls this
service, so the frozen pipeline has exactly one, well-defined seam.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from gnn_data import ATTRIBUTION_DYN, ATTRIBUTION_STATIC
from explain_gnn import ExplainContext
from models.attribution import SourceAttributor
from advisor.config import CONFIG, AdvisorConfig, aqi_band, grap_stage
from advisor.feature_space import get_feature_scaler, get_raw_dynamics, wind_compass


def _r(v):
    """Round to 1 dp, pass through None/NaN safely."""
    if v is None:
        return None
    try:
        v = float(v)
        return None if v != v else round(v, 1)
    except (TypeError, ValueError):
        return None


@dataclass
class Forecast:
    ward_id: str
    node_idx: int
    point_id: int
    time: pd.Timestamp
    horizon_h: int
    p10: float
    p50: float
    p90: float

    def as_dict(self) -> dict:
        return {"predicted_aqi": round(self.p50, 1),
                "aqi_low": round(self.p10, 1), "aqi_high": round(self.p90, 1),
                "forecast_horizon": f"{self.horizon_h}h",
                **aqi_band(self.p50), "grap_stage": grap_stage(self.p50)}


class ForecastService:
    def __init__(self, config: AdvisorConfig = CONFIG):
        self.cfg = config
        self.ctx = ExplainContext(horizon=config.horizon, device=config.device)
        self.d = self.ctx.d
        self.attr = SourceAttributor.load(config.attribution_tag)
        self.fs = get_feature_scaler()
        self.raw = get_raw_dynamics()

        self.static_names = self.d.static_names
        self.dyn_names = self.d.dyn_names
        self._dyn_pos = {c: i for i, c in enumerate(self.dyn_names)}
        self._n_static = len(self.static_names)

        # attribution feature frames (scaled) for arbitrary ward-hours
        self._attr_dyn = pd.read_parquet(
            self.cfg.serving_dir.parent / "gnn_processed" / "dynamic_grid_norm.parquet",
            columns=["point_id", "time"] + list(dict.fromkeys(ATTRIBUTION_DYN)),
        ).set_index(["point_id", "time"]).sort_index()
        self._attr_static = self.d.nodes.set_index("node_idx")[
            list(dict.fromkeys(ATTRIBUTION_STATIC))]

        # RAW ward statics (un-scaled) for human-readable metadata / display
        self._raw_nodes = pd.read_parquet(
            self.cfg.serving_dir.parent / "gnn" / "nodes_static.parquet"
        ).set_index("node_idx")

        self.times = self.d.times

        # ---- optional ensemble (improvement 2.1) -------------------------
        # If member checkpoints exist, forecasts/counterfactuals are averaged
        # across them (better accuracy + calibration). Members share the exact
        # snapshot interface as the explainable model, so nothing else changes.
        # Explainability still uses the single ctx.model (SHAP/GNNExplainer).
        self.ensemble = self._load_ensemble()
        if self.ensemble:
            print(f"[serving] using {len(self.ensemble)}-model ensemble for forecasts")

    def _load_ensemble(self) -> list:
        from pathlib import Path
        from models.gnn_forecast import WardGraphTransformer
        ckpt_dir = Path(__file__).resolve().parent.parent / "models" / "checkpoints"
        ckpts = sorted(ckpt_dir.glob(f"gnn_forecast_h{self.cfg.horizon}_notemporal_ens*.pt"))
        members = []
        for p in ckpts:
            c = torch.load(p, map_location=self.ctx.device, weights_only=False)
            m = WardGraphTransformer(
                n_static=len(self.d.static_names), n_dyn=len(self.d.dyn_names),
                hidden=c["config"]["hidden"], heads=c["config"]["heads"],
                n_layers=c["config"]["layers"], dropout=0.0, edge_dim=4,
                temporal=False, quantiles=c["quantiles"]).to(self.ctx.device)
            m.load_state_dict(c["state_dict"]); m.eval()
            members.append((m, c["y_mu"], c["y_sd"]))
        return members

    # ---- identity / time helpers -----------------------------------------
    def ward_to_node(self, ward_id) -> int:
        return self.ctx.ward_to_node(ward_id)

    def node_meta(self, node_idx: int) -> dict:
        r = self.d.nodes[self.d.nodes["node_idx"] == node_idx].iloc[0]
        raw = self._raw_nodes.loc[node_idx]      # un-scaled statics for display
        def rget(k):
            v = raw.get(k)
            return None if v is None or pd.isna(v) else float(v)
        return {"ward_id": str(r["ward_id"]), "ward_name": str(r["ward_name"]),
                "lat": float(r["ward_lat"]), "lon": float(r["ward_lon"]),
                "point_id": int(r["point_id"]),
                "population": int(rget("population_sum")) if rget("population_sum") is not None else None,
                "population_density": rget("population_density_mean"),
                "vulnerable_sites_3km": rget("vulnerable_sites_3km"),
                "road_capacity_3km": rget("road_capacity_3km"),
                "industry_count_5km": rget("industry_count_5km")}

    def latest_time_index(self) -> int:
        return int(len(self.times) - 1 - self.cfg.horizon)  # last t with a real label window

    def resolve_time(self, time=None, time_index=None) -> int:
        if time_index is not None:
            return int(time_index)
        if time is not None:
            pos = np.where(self.times == np.datetime64(pd.Timestamp(time)))[0]
            if len(pos) == 0:
                raise ValueError(f"time {time} not in dataset")
            return int(pos[0])
        return self.latest_time_index()

    # ---- forecast --------------------------------------------------------
    @torch.no_grad()
    def _forward_all(self, node_x: torch.Tensor, t: int) -> np.ndarray:
        """[N, Q] de-normalised quantiles. Averages the ensemble members if
        loaded (improvement 2.1); otherwise the single explainable model."""
        ea = self.ctx.edge_attr(t)
        if self.ensemble:
            acc = None
            for m, mu, sd in self.ensemble:
                q = m(node_x, self.ctx.edge_index, ea).cpu().numpy() * sd + mu
                acc = q if acc is None else acc + q
            return acc / len(self.ensemble)
        out = self.ctx.model(node_x, self.ctx.edge_index, ea)
        return out.cpu().numpy() * self.ctx.y_sd + self.ctx.y_mu

    @torch.no_grad()
    def _quantiles(self, node_x: torch.Tensor, t: int, node_idx: int) -> np.ndarray:
        return self._forward_all(node_x, t)[node_idx]

    def forecast(self, node_idx: int, t: int) -> Forecast:
        q = self._quantiles(self.ctx.node_x(t), t, node_idx)
        meta = self.node_meta(node_idx)
        return Forecast(ward_id=meta["ward_id"], node_idx=node_idx,
                        point_id=meta["point_id"], time=pd.Timestamp(self.times[t]),
                        horizon_h=self.cfg.horizon,
                        p10=float(q[0]), p50=float(q[self.ctx.qmid]), p90=float(q[-1]))

    @torch.no_grad()
    def forecast_all(self, t: int) -> list[dict]:
        """One graph forward -> every ward's p50 AQI (for the map choropleth)."""
        q = self._forward_all(self.ctx.node_x(t), t)             # [N, Q]
        nodes = self.d.nodes.sort_values("node_idx")
        res = []
        for i, r in enumerate(nodes.itertuples()):
            p50 = float(q[i, self.ctx.qmid])
            res.append({"ward_id": str(r.ward_id), "ward_name": str(r.ward_name),
                        "node_idx": int(r.node_idx), "lat": float(r.ward_lat),
                        "lon": float(r.ward_lon), "aqi": round(p50, 1),
                        **aqi_band(p50), "grap_stage": grap_stage(p50)})
        return res

    @torch.no_grad()
    def layers_all(self, t: int) -> list[dict]:
        """Per-ward multi-metric snapshot powering every map layer in one call:
        predicted AQI, current AQI, forecast change, pollutants, and static
        traffic/industry/green proxies. Reuses one graph forward + raw gather."""
        q = self._forward_all(self.ctx.node_x(t), t)
        nodes = self.d.nodes.sort_values("node_idx")
        res = []
        for i, r in enumerate(nodes.itertuples()):
            pid = int(r.point_id)
            raw = self.raw.at(pid, self.times[t])
            cur = raw.get("aqi")
            pred = float(q[i, self.ctx.qmid])
            stat = self._raw_nodes.loc[int(r.node_idx)]
            res.append({
                "ward_id": str(r.ward_id), "ward_name": str(r.ward_name),
                "node_idx": int(r.node_idx), "lat": float(r.ward_lat), "lon": float(r.ward_lon),
                "aqi": round(pred, 1), "current_aqi": None if cur is None else round(cur, 1),
                "forecast_diff": None if cur is None else round(pred - cur, 1),
                "pm2_5": _r(raw.get("pm2_5")), "pm10": _r(raw.get("pm10")),
                "nitrogen_dioxide": _r(raw.get("nitrogen_dioxide")),
                "construction": _r(raw.get("dust")),
                "wind_speed": _r(raw.get("wind_speed_10m")),
                "wind_dir": _r(raw.get("wind_direction_10m")),
                "traffic": _r(float(stat.get("road_capacity_3km", 0.0))),
                "industry": _r(float(stat.get("industry_count_5km", 0.0))),
                **aqi_band(pred),
            })
        return res

    def source_all(self, t: int) -> dict:
        """Dominant source per ward (for the Source Attribution layer). ~1.3s."""
        out = {}
        for r in self.d.nodes.itertuples():
            prof = self.attr.profile(self._attr_row(int(r.node_idx), t))
            out[str(r.ward_id)] = prof.get("dominant_class", "mixed")
        return out

    def history(self, node_idx: int, t: int, hours: int = 48) -> list[dict]:
        """Recent raw AQI series for the ward's cell (for the trend chart)."""
        pid = int(self.d.nodes.loc[self.d.nodes["node_idx"] == node_idx, "point_id"].iloc[0])
        lo = max(0, t - hours)
        out = []
        for tt in range(lo, t + 1):
            r = self.raw.at(pid, self.times[tt])
            if r.get("aqi") is not None:
                out.append({"time": str(self.times[tt]), "aqi": round(r["aqi"], 1)})
        return out

    # ---- meteorology (raw, human-readable) -------------------------------
    def meteorology(self, node_idx: int, t: int) -> dict:
        pid = int(self.d.nodes.loc[self.d.nodes["node_idx"] == node_idx, "point_id"].iloc[0])
        r = self.raw.at(pid, self.times[t])
        return {"wind_speed": r.get("wind_speed_10m"),
                "wind_direction_deg": r.get("wind_direction_10m"),
                "wind_direction": wind_compass(r.get("wind_direction_10m")),
                "humidity": r.get("relative_humidity_2m"),
                "temperature": r.get("temperature_2m"),
                "boundary_layer_height": r.get("boundary_layer_height"),
                "precipitation": r.get("precipitation")}

    def raw_pollutants(self, node_idx: int, t: int) -> dict:
        pid = int(self.d.nodes.loc[self.d.nodes["node_idx"] == node_idx, "point_id"].iloc[0])
        r = self.raw.at(pid, self.times[t])
        keys = ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide",
                "carbon_monoxide", "ozone", "dust"]
        return {k: r.get(k) for k in keys}

    # ---- source attribution ----------------------------------------------
    def _attr_row(self, node_idx: int, t: int) -> pd.Series:
        pid = int(self.d.nodes.loc[self.d.nodes["node_idx"] == node_idx, "point_id"].iloc[0])
        try:
            dyn = self._attr_dyn.loc[(pid, pd.Timestamp(self.times[t]))]
        except KeyError:
            dyn = pd.Series(0.0, index=self._attr_dyn.columns)
        stat = self._attr_static.loc[node_idx]
        return pd.concat([dyn, stat])

    def attribution(self, node_idx: int, t: int) -> dict:
        return self.attr.profile(self._attr_row(node_idx, t))

    # ---- explanations (reuse SHAP + GNNExplainer) ------------------------
    def shap_local(self, node_idx: int, t: int, top: int = 8) -> list[dict]:
        from explain_gnn import shap_ward
        df, _neigh = shap_ward(self.ctx, node_idx, t)
        return [{"feature": r.feature, "shap_aqi": round(float(r.shap_aqi), 2)}
                for r in df.head(top).itertuples()]

    def gnn_neighbours(self, node_idx: int, t: int, top: int = 6) -> list[dict]:
        from explain_gnn import gnn_explain_ward
        edges_df, _ = gnn_explain_ward(self.ctx, node_idx, t, epochs=120)
        return [{"neighbour_ward": str(r.neighbour_ward), "neighbour_name": str(r.neighbour_name),
                 "importance": round(float(r.importance), 3)}
                for r in edges_df.head(top).itertuples()]

    # ---- counterfactual (Part 10) — edit raw features, re-run frozen GNN --
    @torch.no_grad()
    def counterfactual(self, node_idx: int, t: int, feature_multipliers: dict) -> dict:
        base_x = self.ctx.node_x(t).clone()
        base_q = self._quantiles(base_x, t, node_idx)
        x = base_x.clone()
        applied = {}
        for feat, mult in feature_multipliers.items():
            pos = self._dyn_pos.get(feat)
            if pos is None or not self.fs.known(feat):
                continue
            col = self._n_static + pos
            cur_scaled = float(x[node_idx, col].item())
            raw = self.fs.to_raw(feat, cur_scaled)
            new_raw = max(raw * mult, 0.0)
            x[node_idx, col] = self.fs.to_scaled(feat, new_raw)
            applied[feat] = {"from": round(raw, 2), "to": round(new_raw, 2), "mult": mult}
        new_q = self._quantiles(x, t, node_idx)
        return {"aqi_before": round(float(base_q[self.ctx.qmid]), 1),
                "aqi_after": round(float(new_q[self.ctx.qmid]), 1),
                "delta": round(float(new_q[self.ctx.qmid] - base_q[self.ctx.qmid]), 1),
                "applied_features": applied}


@lru_cache(maxsize=1)
def get_forecast_service() -> ForecastService:
    return ForecastService()


if __name__ == "__main__":
    svc = get_forecast_service()
    node = svc.ctx.ward_to_node("239")
    t = svc.resolve_time(time_index=24477)
    fc = svc.forecast(node, t)
    print("forecast:", fc.as_dict())
    print("meteorology:", svc.meteorology(node, t))
    print("raw pollutants:", svc.raw_pollutants(node, t))
    prof = svc.attribution(node, t)
    print("attribution ranked:", [(r["source"], r["score"]) for r in prof["ranked_sources"]])
    cf = svc.counterfactual(node, t, {"nitrogen_dioxide": 0.75, "carbon_monoxide": 0.80})
    print("counterfactual:", cf)
