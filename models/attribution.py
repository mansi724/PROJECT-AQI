"""
models/attribution.py  —  Engine 2: qualitative source attribution
=================================================================
What this engine IS, and what it deliberately is NOT (see gnn_data.py header):

  IS   : a *qualitative, ranked* source profile per ward-hour — which of
         {dust, traffic, biomass-burning, industry, secondary/combustion} is
         plausibly driving the pollution, with a confidence, validated where we
         have observable proxies (pm25/pm10 ratio, a dust/mixed/combustion class).
  NOT  : quantitative apportionment ("industry = 34%"). No source labels exist,
         6 pollutants is not chemical speciation, so PMF/CMB is off the table.

Two learned heads (LightGBM, trained in train_attribution.py):
  * ratio head   -> predicts the observed pm2.5/pm10 ratio (dust<->combustion axis)
  * class head   -> dust_dominated / mixed / combustion_dominated

Plus transparent RULE-BASED directional signals that need no labels:
  * traffic      : NO2 x road_capacity, boosted at rush hour
  * biomass      : fire_upwind x stubble season x FRP  (Punjab/Haryana transport)
  * industrial   : SO2/NO2 ratio x industry density x EDGAR industry share
  * dust         : AOD + dust + high pm2.5/pm10 ratio
  * secondary    : combustion-class probability + CO/NO2 under stagnation

This module builds the feature frame and serves inference (`SourceAttributor`).
Column lists are imported from gnn_data.py so there is one source of truth.
=================================================================
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_data import ATTRIBUTION_DYN, ATTRIBUTION_STATIC, EMISSION_FEATURES

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "gnn_processed"
CKPT_DIR = BASE / "models" / "checkpoints"

RATIO_TARGET = "pm25_pm10_ratio_obs"
CLASS_TARGET = "source_class_obs"
CLASSES = ["dust_dominated", "mixed", "combustion_dominated"]


def build_attribution_frame() -> tuple[pd.DataFrame, list]:
    """Join station labels with their cell dynamics + ward statics.

    Returns (frame, feature_names). Dynamic attribution features come from the
    (scaled) dynamic grid — LightGBM is invariant to the per-feature monotone
    scaling, so using the processed file is safe and avoids loading the 5 GB raw.
    """
    lab = pd.read_parquet(PROC / "labels_station_clean.parquet",
                          columns=["node_idx", "point_id", "time", "split_lab",
                                   RATIO_TARGET, CLASS_TARGET])
    dyn_cols = list(dict.fromkeys(ATTRIBUTION_DYN))
    dyn = pd.read_parquet(PROC / "dynamic_grid_norm.parquet",
                          columns=["point_id", "time"] + dyn_cols)
    stat_cols = list(dict.fromkeys(ATTRIBUTION_STATIC))
    nod = pd.read_parquet(PROC / "nodes_static_norm.parquet",
                          columns=["node_idx"] + stat_cols)

    df = lab.merge(dyn, on=["point_id", "time"], how="left") \
            .merge(nod, on="node_idx", how="left")
    feats = dyn_cols + stat_cols
    return df, feats


class SourceAttributor:
    """Inference: turn one ward-hour feature row into a ranked source profile.

    Numbers stay qualitative — scores are relative plausibilities in [0,1], never
    a mass %.  Engine 3 (future) will cite these, not recompute them.
    """

    SIGNALS = ["dust", "traffic", "biomass_burning", "industrial", "secondary"]

    def __init__(self, ratio_model=None, class_model=None, feat_names=None):
        self.ratio_model = ratio_model
        self.class_model = class_model
        self.feat_names = feat_names

    @classmethod
    def load(cls, horizon_tag: str = "attribution"):
        with open(CKPT_DIR / f"{horizon_tag}_ratio.pkl", "rb") as f:
            ratio = pickle.load(f)
        with open(CKPT_DIR / f"{horizon_tag}_class.pkl", "rb") as f:
            clazz = pickle.load(f)
        with open(CKPT_DIR / f"{horizon_tag}_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        return cls(ratio, clazz, meta["features"])

    # ---- rule-based directional signals (label-free, transparent) --------
    @staticmethod
    def _norm(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo + 1e-9), 0, 1))

    def rule_signals(self, row: pd.Series) -> dict:
        g = lambda k, d=0.0: float(row[k]) if k in row and pd.notna(row[k]) else d
        # values are scaled; use sign/relative magnitude thresholds around 0 (z-space)
        traffic = self._norm(g("nitrogen_dioxide"), -0.5, 2.0) * (0.6 + 0.4 * g("is_rush_hour")) \
            * (0.5 + 0.5 * self._norm(g("road_capacity_3km"), -0.5, 2.0))
        biomass = g("fire_upwind") * (0.5 + 0.5 * g("is_stubble_season")) \
            * (0.4 + 0.6 * self._norm(g("fire_frp_sum"), -0.3, 2.0))
        industrial = self._norm(g("so2_no2_ratio"), -0.5, 2.0) \
            * (0.4 + 0.6 * self._norm(g("industry_count_5km"), -0.5, 2.0)) \
            * (0.5 + 0.5 * self._norm(g("emis_industry_pm25"), -0.5, 2.0))
        dust = 0.5 * self._norm(g("dust"), -0.3, 2.0) + 0.5 * self._norm(g("aerosol_optical_depth"), -0.3, 2.0)
        secondary = self._norm(g("carbon_monoxide"), -0.5, 2.0) * (0.4 + 0.6 * self._norm(g("stagnation_index"), -0.5, 2.0))
        return {"dust": dust, "traffic": traffic, "biomass_burning": biomass,
                "industrial": industrial, "secondary": secondary}

    def edgar_prior(self, row: pd.Series) -> dict:
        """Per-ward EDGAR sector shares of PM2.5 (annual) — context prior only."""
        sectors = ["power", "industry", "residential", "transport"]
        vals = {s: max(float(row.get(f"emis_{s}_pm25", 0.0)), 0.0) for s in sectors}
        tot = sum(vals.values()) + 1e-9
        return {s: vals[s] / tot for s in sectors}

    def profile(self, row: pd.Series) -> dict:
        """Ranked qualitative source profile for one ward-hour."""
        X = np.array([[float(row.get(f, 0.0)) for f in self.feat_names]])
        out = {}
        if self.ratio_model is not None:
            out["pred_pm25_pm10_ratio"] = float(self.ratio_model.predict(X)[0])
        if self.class_model is not None:
            proba = self.class_model.predict_proba(X)[0]
            classes = list(self.class_model.classes_)
            out["class_probs"] = {c: float(p) for c, p in zip(classes, proba)}
            out["dominant_class"] = classes[int(np.argmax(proba))]

        signals = self.rule_signals(row)
        # fold the learned class posterior into the matching signals
        cp = out.get("class_probs", {})
        signals["dust"] = 0.6 * signals["dust"] + 0.4 * cp.get("dust_dominated", 0.0)
        signals["secondary"] = 0.6 * signals["secondary"] + 0.4 * cp.get("combustion_dominated", 0.0)

        s = sum(signals.values()) + 1e-9
        ranked = sorted(((k, v / s) for k, v in signals.items()), key=lambda kv: -kv[1])
        out["signals"] = signals
        out["ranked_sources"] = [{"source": k, "score": round(v, 3)} for k, v in ranked]
        out["confidence"] = round(float(max(cp.values())) if cp else 0.5, 3)
        # 4.4 — per-attribution uncertainty: normalised entropy of the class
        # posterior (0 = certain, 1 = maximally uncertain) + top-2 margin.
        if cp:
            p = np.array(list(cp.values()), dtype=float)
            p = p[p > 0]
            ent = float(-(p * np.log(p)).sum() / np.log(len(cp))) if len(cp) > 1 else 0.0
            top2 = sorted(cp.values(), reverse=True)[:2]
            out["uncertainty"] = round(ent, 3)
            out["margin"] = round(float(top2[0] - (top2[1] if len(top2) > 1 else 0.0)), 3)
        else:
            out["uncertainty"], out["margin"] = 0.5, 0.0
        out["edgar_prior_pm25"] = {k: round(v, 3) for k, v in self.edgar_prior(row).items()}
        return out


if __name__ == "__main__":
    df, feats = build_attribution_frame()
    print("attribution frame:", df.shape, "| features:", len(feats))
    print("nulls in targets:", df[RATIO_TARGET].isna().mean().round(3),
          df[CLASS_TARGET].isna().mean().round(3))
    print("split_lab:", df["split_lab"].value_counts().to_dict())
