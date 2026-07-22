"""
gnn_data.py
=================================================================
Loader for the GNN dataset. This is the ONLY thing training code should import.

The central idea
----------------
The old flat table stored 7,601,856 ward-hour rows, but the dynamic data has
only 499,776 unique (cell, hour) records — every pollutant/weather column is
identical for all wards in a CAMS cell. We keep the dynamic tensor at its TRUE
resolution [T, 19, F] (~120 MB) and broadcast to wards **on the fly** with an
index array. Materialising [T, 289, F] would cost ~1.8 GB for zero extra
information, which is exactly the mistake the old file made on disk.

    X_dyn[t][cell_of_node]  ->  [289, F_dyn]     # free, it's a gather

Shapes
------
    edge_index   [2, E]        int64   ward graph (queen contiguity)
    edge_attr    [E, 2]        float32 (dist_km, bearing_deg)
    X_static     [N, F_s]      float32 per-ward, time-invariant
    X_dyn        [T, C, F_d]   float32 per-cell, per-hour
    cell_of_node [N]           int64   ward -> cell gather index
    y_grid       [T, C]        float32 CAMS-derived target (dense, weak)
    y_station    sparse        real CPCB AQI (see labels_station.parquet)

Usage
-----
    from src.data.gnn_data import load_gnn
    d = load_gnn(horizon=24)
    x_t = d.node_features(t)          # [N, F_s + F_d] at timestep t
    seq = d.window(t, lookback=24)    # [24, N, F] for a spatio-temporal model
=================================================================
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
GNN = BASE / "data" / "gnn"

# ---- feature groups -------------------------------------------------------
# Static, genuinely per-ward (recomputed on ward geometry — NOT grid copies).
STATIC_FEATURES = [
    "ward_area_km2", "road_km_3km", "road_capacity_3km", "road_km_in_ward",
    "road_capacity_in_ward", "road_km_per_km2", "industry_count_5km",
    "industry_count_in_ward", "vulnerable_sites_3km", "vulnerable_sites_in_ward",
    "population_sum", "population_density_mean", "elevation_mean",
    "lu_builtup_fraction", "lu_tree_fraction",
]

# Dynamic, per grid cell per hour.
DYN_CURRENT = [
    "pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide",
    "ozone", "aerosol_optical_depth", "dust", "temperature_2m",
    "relative_humidity_2m", "dew_point_2m", "precipitation", "surface_pressure",
    "wind_speed_10m", "wind_gusts_10m", "boundary_layer_height",
    "fire_count", "fire_frp_sum", "aqi",
]
DYN_DERIVED = [
    "wind_dir_sin", "wind_dir_cos", "ventilation_index", "stagnation_index",
    "wind_from_nw", "hour_sin", "hour_cos", "month_sin", "month_cos",
    "dow_sin", "dow_cos", "is_weekend", "is_rush_hour", "is_winter",
    "is_stubble_season", "is_diwali_window",
]

# Physical source-signature features. `pm25_pm10_ratio` has a MEASURED twin in
# labels_station (`pm25_pm10_ratio_obs`), so predicting it is a scoreable task.
DYN_SIGNATURE = [
    "pm25_pm10_ratio", "dust_fraction", "fire_upwind", "so2_no2_ratio",
]
DYN_HISTORY = [
    *[f"aqi_lag_{h}h" for h in (1, 3, 6, 12, 24, 48)],
    *[f"pm2_5_lag_{h}h" for h in (1, 3, 6, 12, 24, 48)],
    *[f"pm10_lag_{h}h" for h in (1, 3, 6, 12, 24, 48)],
    *[f"nitrogen_dioxide_lag_{h}h" for h in (1, 3, 6, 12, 24, 48)],
    "aqi_roll_mean_6h", "aqi_roll_mean_24h", "aqi_roll_max_24h", "aqi_roll_std_24h",
    "pm2_5_roll_mean_6h", "pm2_5_roll_mean_24h", "pm2_5_roll_max_24h", "pm2_5_roll_std_24h",
    "aqi_diff_1h", "aqi_diff_24h",
]

# NEVER feed these — they are the answer or a pre-correction copy.
LEAK = {"target_aqi_t24", "target_aqi_t48", "target_aqi_t72",
        "persistence_aqi_t24", "pm2_5_raw", "pm10_raw",
        "aqi_station", "dominant_pollutant_station",
        "pm25_pm10_ratio_obs", "source_class_obs"}

# Engine 2 (attribution) reads these off nodes_static; they are annual EDGAR
# constants, so they carry no temporal signal and are excluded from forecasting.
EMISSION_FEATURES = [
    f"emis_{s}_{p}" for p in ("pm25", "nox", "so2", "co")
    for s in ("power", "industry", "residential", "transport", "total")
]

# ---- Engine 2: source attribution ----------------------------------------
# What is defensible here, and what is not:
#   CAN   dust-vs-combustion split      -> validate against pm25_pm10_ratio_obs
#         traffic signal                -> NO2 + real per-ward road_capacity_3km
#         stubble transport             -> fire_upwind (FIRMS FRP gated on NW wind)
#         regional transport            -> wind-gated edges, cos(bearing - wind_dir)
#         sector context                -> EDGAR shares, as a PRIOR only
#   CANNOT quantitative apportionment ("industry = 34% of today's PM2.5").
#         No source labels exist, and 6 pollutants is not chemical speciation,
#         so PMF/CMB receptor modelling is not possible. Do not claim it.
ATTRIBUTION_DYN = DYN_SIGNATURE + [
    "pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide",
    "ozone", "dust", "fire_count", "fire_frp_sum", "wind_speed_10m",
    "wind_dir_sin", "wind_dir_cos", "wind_from_nw", "stagnation_index",
    "ventilation_index", "is_stubble_season", "is_winter",
]
ATTRIBUTION_STATIC = [
    "road_capacity_3km", "road_km_per_km2", "industry_count_5km",
    "industry_count_in_ward", "lu_builtup_fraction", "lu_tree_fraction",
    "population_density_mean",
] + EMISSION_FEATURES


@dataclass
class GNNData:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    times: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    X_static: np.ndarray
    X_dyn: np.ndarray
    cell_of_node: np.ndarray
    y_grid: np.ndarray
    split_of_t: np.ndarray
    static_names: list
    dyn_names: list
    labels: pd.DataFrame = field(default=None)

    @property
    def n_nodes(self): return self.X_static.shape[0]

    @property
    def n_times(self): return self.X_dyn.shape[0]

    def node_features(self, t: int) -> np.ndarray:
        """[N, F_s + F_d] — the gather is what keeps this cheap."""
        return np.concatenate([self.X_static, self.X_dyn[t][self.cell_of_node]], axis=1)

    def window(self, t: int, lookback: int = 24) -> np.ndarray:
        """[lookback, N, F] ending at t (inclusive)."""
        lo = max(0, t - lookback + 1)
        return np.stack([self.node_features(i) for i in range(lo, t + 1)])

    def targets(self, t: int) -> np.ndarray:
        """[N] CAMS-derived target broadcast to wards (weak label — see docs)."""
        return self.y_grid[t][self.cell_of_node]

    def t_indices(self, split: str) -> np.ndarray:
        return np.where(self.split_of_t == split)[0]

    def feature_names(self) -> list:
        return self.static_names + self.dyn_names


def load_gnn(horizon: int = 24, with_labels: bool = True,
             dyn_features: list = None) -> GNNData:
    nodes = pd.read_parquet(GNN / "nodes_static.parquet").sort_values("node_idx")
    edges = pd.read_parquet(GNN / "edges.parquet")
    dyn = pd.read_parquet(GNN / "dynamic_grid.parquet")

    dyn_names = dyn_features or (DYN_CURRENT + DYN_DERIVED + DYN_SIGNATURE + DYN_HISTORY)
    dyn_names = [c for c in dyn_names if c in dyn.columns and c not in LEAK]
    static_names = [c for c in STATIC_FEATURES if c in nodes.columns]

    # ---- pivot dynamic to [T, C, F] -------------------------------------
    cells = np.sort(dyn["point_id"].unique())
    times = np.sort(dyn["time"].unique())
    cell_pos = {c: i for i, c in enumerate(cells)}
    time_pos = {t: i for i, t in enumerate(times)}
    dyn = dyn.sort_values(["time", "point_id"])
    ci = dyn["point_id"].map(cell_pos).to_numpy()
    ti = dyn["time"].map(time_pos).to_numpy()

    X_dyn = np.full((len(times), len(cells), len(dyn_names)), np.nan, dtype="float32")
    X_dyn[ti, ci] = dyn[dyn_names].to_numpy(dtype="float32")

    tgt = f"target_aqi_t{horizon}"
    y_grid = np.full((len(times), len(cells)), np.nan, dtype="float32")
    y_grid[ti, ci] = dyn[tgt].to_numpy(dtype="float32")

    split_of_t = np.empty(len(times), dtype=object)
    split_of_t[ti] = dyn["split"].to_numpy()

    # ---- ward -> cell gather --------------------------------------------
    if nodes["point_id"].isna().any():
        raise ValueError("nodes_static has wards with no point_id")
    cell_of_node = nodes["point_id"].map(cell_pos).to_numpy(dtype="int64")
    X_static = nodes[static_names].to_numpy(dtype="float32")

    # ---- graph -----------------------------------------------------------
    edge_index = np.stack([edges["src"].to_numpy(), edges["dst"].to_numpy()]).astype("int64")
    edge_attr = edges[["dist_km", "bearing_deg"]].to_numpy(dtype="float32")

    labels = None
    if with_labels:
        p = GNN / "labels_station.parquet"
        if p.exists():
            labels = pd.read_parquet(p)
            labels["t_idx"] = labels["time"].map(time_pos)
            labels = labels[labels["t_idx"].notna() & labels["node_idx"].notna()]
            labels["t_idx"] = labels["t_idx"].astype(int)

    return GNNData(nodes=nodes, edges=edges, times=times, edge_index=edge_index,
                   edge_attr=edge_attr, X_static=X_static, X_dyn=X_dyn,
                   cell_of_node=cell_of_node, y_grid=y_grid, split_of_t=split_of_t,
                   static_names=static_names, dyn_names=dyn_names, labels=labels)


def to_torch(d: GNNData):
    """Convert to torch tensors (edge_index ready for PyG)."""
    import torch
    return {
        "edge_index": torch.from_numpy(d.edge_index),
        "edge_attr": torch.from_numpy(d.edge_attr),
        "X_static": torch.from_numpy(d.X_static),
        "X_dyn": torch.from_numpy(d.X_dyn),
        "cell_of_node": torch.from_numpy(d.cell_of_node),
        "y_grid": torch.from_numpy(d.y_grid),
    }


if __name__ == "__main__":
    d = load_gnn()
    print(f"nodes        : {d.n_nodes}")
    print(f"timesteps    : {d.n_times}")
    print(f"edges        : {d.edge_index.shape[1]} (mean degree {d.edge_index.shape[1]/d.n_nodes:.1f})")
    print(f"X_static     : {d.X_static.shape}  {d.X_static.nbytes/1e6:.1f} MB")
    print(f"X_dyn        : {d.X_dyn.shape}  {d.X_dyn.nbytes/1e6:.1f} MB")
    print(f"features/node: {len(d.feature_names())}")
    print(f"split t      : train={len(d.t_indices('train'))} val={len(d.t_indices('val'))} test={len(d.t_indices('test'))}")
    x = d.node_features(1000)
    print(f"node_features(1000) -> {x.shape}")
    if d.labels is not None:
        print(f"labels       : {len(d.labels):,} station-hours, {d.labels['ward_id'].nunique()} wards")
    else:
        print("labels       : not built yet (run build_gnn_dataset.py --labels)")
    mat = (d.X_dyn.nbytes + d.X_static.nbytes) / 1e6
    print(f"\ntotal in-memory: {mat:.0f} MB  (old flat table: ~5300 MB)")
