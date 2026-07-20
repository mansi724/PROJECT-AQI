"""
advisor/feature_space.py — raw <-> scaled feature transforms + raw display values.

Reuses the EXACT scalers the preprocessing step fitted (`scalers.joblib`), so a
counterfactual edit made in human-readable units (e.g. "cut NO2 by 25%") maps to
the identical z-space the frozen GNN was trained on. Nothing is re-fitted; we
only read the per-column parameters off the already-fitted sklearn objects.

Two services:
  * `FeatureScaler` — `to_scaled(col, raw)` / `to_raw(col, scaled)` per column,
    honouring each column's group transform (Robust / Standard / log1p+Standard /
    as-is).
  * `RawDynamics`   — the raw (un-scaled) pollutant/weather values per cell-hour,
    read straight from `data/gnn/dynamic_grid.parquet`, for display + as the base
    values a counterfactual multiplies.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "gnn_processed"
RAW_DYN = BASE / "data" / "gnn" / "dynamic_grid.parquet"

# groups(cols) key  ->  scalers(fitted object) key
_GROUP_ALIAS = {
    "dyn_robust": "dyn_robust",
    "dyn_standard": "dyn_standard",
    "dyn_log1p": "dyn_log1p_standard",
    "stat_log1p": "static_log1p_standard",
    "stat_standard": "static_standard",
}
_LOG1P_GROUPS = {"dyn_log1p", "stat_log1p"}


class FeatureScaler:
    """Per-column raw<->scaled transform using the fitted preprocessing scalers."""

    def __init__(self, scalers_path: Path = PROC / "scalers.joblib"):
        bundle = joblib.load(scalers_path)
        scalers, groups = bundle["scalers"], bundle["groups"]
        # col -> (kind, sklearn_obj, index_in_group, is_log1p) ; None => identity
        self._map: dict[str, tuple | None] = {}
        for gname, cols in groups.items():
            if gname == "dyn_asis" or gname not in _GROUP_ALIAS:
                for c in cols:
                    self._map[c] = None                 # identity (sin/cos, flags)
                continue
            entry = scalers[_GROUP_ALIAS[gname]]
            obj = entry["scaler"] if isinstance(entry, dict) else entry
            kind = type(obj).__name__
            is_log = gname in _LOG1P_GROUPS
            for i, c in enumerate(cols):
                self._map[c] = (kind, obj, i, is_log)

    def _params(self, spec):
        kind, obj, i, is_log = spec
        if kind == "RobustScaler":
            center, scale = float(obj.center_[i]), float(obj.scale_[i])
        else:  # StandardScaler
            center, scale = float(obj.mean_[i]), float(obj.scale_[i])
        return center, scale, is_log

    def to_scaled(self, col: str, raw: float) -> float:
        spec = self._map.get(col)
        if spec is None:
            return float(raw)
        center, scale, is_log = self._params(spec)
        x = np.log1p(raw) if is_log else raw
        return float((x - center) / (scale if scale else 1.0))

    def to_raw(self, col: str, scaled: float) -> float:
        spec = self._map.get(col)
        if spec is None:
            return float(scaled)
        center, scale, is_log = self._params(spec)
        x = scaled * (scale if scale else 1.0) + center
        return float(np.expm1(x) if is_log else x)

    def known(self, col: str) -> bool:
        return col in self._map


class RawDynamics:
    """Raw (un-scaled) pollutant/weather values per (point_id, time)."""

    def __init__(self, path: Path = RAW_DYN):
        cols = ["point_id", "time", "aqi", "pm2_5", "pm10", "nitrogen_dioxide",
                "sulphur_dioxide", "carbon_monoxide", "ozone", "dust",
                "aerosol_optical_depth", "temperature_2m", "relative_humidity_2m",
                "dew_point_2m", "wind_speed_10m", "wind_direction_10m",
                "wind_gusts_10m", "surface_pressure", "boundary_layer_height",
                "precipitation", "fire_count", "fire_frp_sum"]
        df = pd.read_parquet(path, columns=cols)
        self.df = df.set_index(["point_id", "time"]).sort_index()

    def at(self, point_id: int, time) -> dict:
        try:
            row = self.df.loc[(int(point_id), pd.Timestamp(time))]
        except KeyError:
            return {}
        return {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}


_WIND_16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def wind_compass(deg: float | None) -> str | None:
    if deg is None or (isinstance(deg, float) and np.isnan(deg)):
        return None
    return _WIND_16[int((deg % 360) / 22.5 + 0.5) % 16]


@lru_cache(maxsize=1)
def get_feature_scaler() -> FeatureScaler:
    return FeatureScaler()


@lru_cache(maxsize=1)
def get_raw_dynamics() -> RawDynamics:
    return RawDynamics()


if __name__ == "__main__":
    fs = get_feature_scaler()
    # round-trip a few columns
    for col, raw in [("nitrogen_dioxide", 80.0), ("pm2_5", 250.0),
                     ("wind_speed_10m", 2.4), ("fire_count", 12.0), ("hour_sin", 0.5)]:
        s = fs.to_scaled(col, raw)
        back = fs.to_raw(col, s)
        print(f"{col:20} raw={raw:8.2f} -> scaled={s:7.3f} -> raw={back:8.2f}  "
              f"{'OK' if abs(back-raw) < 1e-3 else 'MISMATCH'}")
    rd = get_raw_dynamics()
    import numpy as np
    t = rd.df.index.get_level_values(1).max()
    pid = rd.df.index.get_level_values(0)[0]
    print("\nraw dynamics sample @", pid, str(t)[:16], "->",
          {k: rd.at(pid, t).get(k) for k in ["aqi", "wind_speed_10m", "wind_direction_10m", "relative_humidity_2m"]})
    print("wind 333 ->", wind_compass(333), "| 45 ->", wind_compass(45))
