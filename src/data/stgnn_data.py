"""
stgnn_data.py  —  model-facing loader for the PROCESSED GNN dataset
=================================================================
Relationship to the other loaders (read this to avoid the double-scale trap):

    gnn_data.py       loads data/gnn/            (RAW, unnormalised)  -> analysis / GBDT
    stgnn_data.py     loads data/gnn_processed/  (already scaled)     -> the neural net  <-- HERE

PREPROCESSING.md is explicit: the STGNN reads `data/gnn_processed/` and **skips
re-normalisation** (the scalers were fit on TRAIN hours only). Feeding the raw
`data/gnn/` tensors to a neural net would train on unscaled inputs — that is the
mistake this module exists to prevent.

Single source of truth for columns
-----------------------------------
The feature-group / leak lists live in ONE place — `gnn_data.py` — and are
imported here. We do NOT redefine them, so the raw and processed loaders can
never drift apart. Only the *paths* and the *label filename*
(`labels_station_clean.parquet`) differ.

Memory design (unchanged from gnn_data.py, and it matters)
----------------------------------------------------------
Dynamic data has only 19 distinct (cell, hour) series. Keep X_dyn at its true
resolution [T, 19, F] (~140 MB) and gather to the 289 wards on the fly:

    X_dyn[t][cell_of_node]  ->  [289, F_dyn]      # a free gather, never stored

Temporal models get an extra win: run the GRU on the 19 cell sequences, not on
289 duplicated ward sequences, then gather the encoded state to wards.

Targets & labels are RAW (AQI points), never scaled — so `metrics.py` scores the
GNN and the GBDT baseline in the same units.
=================================================================
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the column groups — do not redefine (single source of truth).
from src.data.gnn_data import (
    STATIC_FEATURES, DYN_CURRENT, DYN_DERIVED, DYN_SIGNATURE, DYN_HISTORY, LEAK,
)

BASE = Path(__file__).resolve().parents[2]
PROC = BASE / "data" / "gnn_processed"
REALTIME = BASE / "data" / "realtime"

DYN_DEFAULT = DYN_CURRENT + DYN_DERIVED + DYN_SIGNATURE + DYN_HISTORY


def _realtime_on(flag: bool | None) -> bool:
    """Live serving is opt-in: explicit arg wins, else the AQI_REALTIME env flag.
    Training paths never pass the flag, so they stay on the frozen historical set."""
    if flag is not None:
        return flag
    return os.getenv("AQI_REALTIME", "0") == "1"


def _append_realtime(dyn: pd.DataFrame) -> pd.DataFrame:
    """Concat the live rolling window onto the historical dynamic frame, letting
    live rows win on any overlapping (point_id, time). Missing/extra columns are
    reconciled to the historical schema so the downstream pivot is unaffected."""
    rt_path = REALTIME / "dynamic_grid_norm.parquet"
    if not rt_path.exists():
        return dyn
    rt = pd.read_parquet(rt_path)
    rt = rt.reindex(columns=dyn.columns)      # align to historical schema
    both = pd.concat([dyn, rt], ignore_index=True)
    both = both.drop_duplicates(["point_id", "time"], keep="last").reset_index(drop=True)
    return both


@dataclass
class STGNNData:
    # frames (kept for names / joins / geometry)
    nodes: pd.DataFrame
    edges: pd.DataFrame
    labels: pd.DataFrame                 # station AQI, with t_idx attached
    times: np.ndarray                    # [T] datetime64, sorted
    # graph
    edge_index: np.ndarray               # [2, E] int64, undirected (both dirs stored)
    edge_attr: np.ndarray                # [E, 3] float32 (dist_km_z, bearing_sin, bearing_cos)
    edge_bearing_deg: np.ndarray         # [E] float32 raw bearing, for the wind gate
    # features
    X_static: np.ndarray                 # [N, F_s] float32, already scaled
    X_dyn: np.ndarray                    # [T, C, F_d] float32, already scaled
    cell_of_node: np.ndarray             # [N] int64 ward -> cell position
    # targets / splits (RAW AQI points, never scaled)
    y_grid: np.ndarray                   # [T, C] target_aqi_t{h}, cell-level
    persist_grid: np.ndarray             # [T, C] persistence_aqi_t24 (naive baseline)
    split_of_t: np.ndarray               # [T] 'train'|'val'|'test' (full 3-yr chrono split)
    # meta
    static_names: list = field(default_factory=list)
    dyn_names: list = field(default_factory=list)
    horizon: int = 24
    # wind index for the gate (position of each dyn col), filled in load
    _wind_sin_col: int = -1
    _wind_cos_col: int = -1

    @property
    def n_nodes(self): return self.X_static.shape[0]

    @property
    def n_times(self): return self.X_dyn.shape[0]

    @property
    def n_cells(self): return self.X_dyn.shape[1]

    def node_features(self, t: int) -> np.ndarray:
        """[N, F_s + F_d] at timestep t — the gather is what keeps this cheap."""
        return np.concatenate([self.X_static, self.X_dyn[t][self.cell_of_node]], axis=1)

    def window(self, t: int, lookback: int = 24) -> np.ndarray:
        """[lookback, C, F_d] of cell-level dynamics ending at t (inclusive).

        Cell-resolution on purpose: encode 19 sequences, then gather to wards.
        """
        lo = max(0, t - lookback + 1)
        seq = self.X_dyn[lo:t + 1]                       # [<=L, C, F_d]
        if seq.shape[0] < lookback:                      # left-pad by repeating first row
            pad = np.repeat(seq[:1], lookback - seq.shape[0], axis=0)
            seq = np.concatenate([pad, seq], axis=0)
        return seq

    def t_indices(self, split: str) -> np.ndarray:
        return np.where(self.split_of_t == split)[0]

    def feature_names(self) -> list:
        return self.static_names + self.dyn_names

    # ---- wind gate helper -------------------------------------------------
    def wind_alignment(self, t: int) -> np.ndarray:
        """[E] cos(bearing_src->dst - wind_dir) using the SOURCE cell's wind.

        +1 => edge points exactly downwind of the source (pollution transported
        src->dst); -1 => upwind. Built from sin/cos so there is no degree wrap.
        Convention note: `wind_direction_10m` is the direction the wind blows
        *from*; the model is free to learn the sign, so we expose the raw
        alignment and let attention decide.
        """
        src_cells = self.cell_of_node[self.edge_index[0]]
        w_sin = self.X_dyn[t, src_cells, self._wind_sin_col]
        w_cos = self.X_dyn[t, src_cells, self._wind_cos_col]
        b = np.deg2rad(self.edge_bearing_deg)
        return (np.cos(b) * w_cos + np.sin(b) * w_sin).astype("float32")


def load_stgnn(horizon: int = 24, dyn_features: list | None = None,
               realtime: bool | None = None) -> STGNNData:
    nodes = pd.read_parquet(PROC / "nodes_static_norm.parquet").sort_values("node_idx").reset_index(drop=True)
    edges = pd.read_parquet(PROC / "edges_norm.parquet")
    dyn = pd.read_parquet(PROC / "dynamic_grid_norm.parquet")
    labels = pd.read_parquet(PROC / "labels_station_clean.parquet")

    if _realtime_on(realtime):
        dyn = _append_realtime(dyn)

    dyn_names = [c for c in (dyn_features or DYN_DEFAULT) if c in dyn.columns and c not in LEAK]
    static_names = [c for c in STATIC_FEATURES if c in nodes.columns]

    # ---- pivot dynamic to [T, C, F] --------------------------------------
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

    persist_grid = np.full((len(times), len(cells)), np.nan, dtype="float32")
    persist_grid[ti, ci] = dyn["persistence_aqi_t24"].to_numpy(dtype="float32")

    split_of_t = np.empty(len(times), dtype=object)
    split_of_t[ti] = dyn["split"].to_numpy()

    # ---- static + ward->cell gather --------------------------------------
    if nodes["point_id"].isna().any():
        raise ValueError("nodes_static_norm has wards with no point_id")
    cell_of_node = nodes["point_id"].map(cell_pos).to_numpy(dtype="int64")
    X_static = nodes[static_names].to_numpy(dtype="float32")

    # ---- graph (edges already undirected: both directions stored) --------
    edge_index = np.stack([edges["src"].to_numpy(), edges["dst"].to_numpy()]).astype("int64")
    edge_attr = edges[["dist_km_z", "bearing_sin", "bearing_cos"]].to_numpy(dtype="float32")
    edge_bearing_deg = edges["bearing_deg"].to_numpy(dtype="float32")

    # ---- labels: attach t_idx, keep valid rows ---------------------------
    labels = labels.copy()
    labels["t_idx"] = labels["time"].map(time_pos)
    labels = labels[labels["t_idx"].notna() & labels["node_idx"].notna()].copy()
    labels["t_idx"] = labels["t_idx"].astype(int)
    labels["node_idx"] = labels["node_idx"].astype(int)

    d = STGNNData(
        nodes=nodes, edges=edges, labels=labels, times=times,
        edge_index=edge_index, edge_attr=edge_attr, edge_bearing_deg=edge_bearing_deg,
        X_static=X_static, X_dyn=X_dyn, cell_of_node=cell_of_node,
        y_grid=y_grid, persist_grid=persist_grid, split_of_t=split_of_t,
        static_names=static_names, dyn_names=dyn_names, horizon=horizon,
    )
    d._wind_sin_col = dyn_names.index("wind_dir_sin")
    d._wind_cos_col = dyn_names.index("wind_dir_cos")
    return d


def build_supervised_pairs(d: STGNNData, split_lab: str):
    """Form (input_t, node, label, persistence) tuples for the station task.

    A supervised example is: inputs at time `t_input`, target = the station's
    AQI `horizon` hours later. Persistence = the station's OWN reading at
    t_input (true 'tomorrow = today'); falls back to the cell CAMS AQI when the
    station has no reading at t_input.

    Returns dict of int64/float32 arrays: t_input, node, y, y_persist, cell.
    The `split_lab` column (labelled-era chronological split) gates which rows
    are eligible — never random-split.
    """
    lab = d.labels
    h = d.horizon

    # station reading indexed by (node, t) for label lookup and persistence
    obs = {(n, t): v for n, t, v in
           zip(lab["node_idx"].to_numpy(), lab["t_idx"].to_numpy(),
               lab["aqi_station"].to_numpy())}

    rows = lab[lab["split_lab"] == split_lab]
    t_in, node, y, y_p, cell = [], [], [], [], []
    for n, t_lab, aqi in zip(rows["node_idx"].to_numpy(),
                             rows["t_idx"].to_numpy(),
                             rows["aqi_station"].to_numpy()):
        t0 = t_lab - h                      # inputs are `h` hours before the label
        if t0 < 0:
            continue
        c = d.cell_of_node[n]
        p = obs.get((n, t0))
        if p is None:
            p = d.persist_grid[t0, c]       # fallback: cell CAMS AQI at t0
        if not np.isfinite(p):
            continue
        t_in.append(t0); node.append(n); y.append(aqi); y_p.append(p); cell.append(c)

    return {
        "t_input": np.asarray(t_in, dtype="int64"),
        "node": np.asarray(node, dtype="int64"),
        "y": np.asarray(y, dtype="float32"),
        "y_persist": np.asarray(y_p, dtype="float32"),
        "cell": np.asarray(cell, dtype="int64"),
    }


if __name__ == "__main__":
    d = load_stgnn(horizon=24)
    print(f"nodes         : {d.n_nodes}")
    print(f"cells (point) : {d.n_cells}")
    print(f"timesteps     : {d.n_times}   {str(d.times.min())[:16]} -> {str(d.times.max())[:16]}")
    print(f"edges         : {d.edge_index.shape[1]} (mean degree {d.edge_index.shape[1]/d.n_nodes:.1f})")
    print(f"X_static      : {d.X_static.shape}  {d.X_static.nbytes/1e6:.1f} MB")
    print(f"X_dyn         : {d.X_dyn.shape}  {d.X_dyn.nbytes/1e6:.1f} MB")
    print(f"dyn features  : {len(d.dyn_names)} | static features : {len(d.static_names)}")
    print(f"split t       : train={len(d.t_indices('train'))} val={len(d.t_indices('val'))} test={len(d.t_indices('test'))}")
    print(f"NaN in X_dyn  : {np.isnan(d.X_dyn).sum()}  | NaN in X_static: {np.isnan(d.X_static).sum()}")
    print(f"labelled wards: {d.labels['node_idx'].nunique()}  station-hours: {len(d.labels):,}")
    for sp in ("train", "val", "test"):
        pairs = build_supervised_pairs(d, sp)
        print(f"  supervised[{sp:>5}]: {len(pairs['y']):>6,} pairs  "
              f"y[{pairs['y'].min():.0f},{pairs['y'].max():.0f}]  "
              f"persist RMSE={np.sqrt(np.mean((pairs['y']-pairs['y_persist'])**2)):.2f}")
    wa = d.wind_alignment(1000)
    print(f"wind_alignment(1000): [{wa.min():.2f},{wa.max():.2f}] mean {wa.mean():.2f}")
