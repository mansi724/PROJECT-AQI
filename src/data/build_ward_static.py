"""
build_ward_static.py
=================================================================
Builds the GNN NODE table and EDGE table.

Why this exists
---------------
`build_dataset.py` computed the OSM features (roads / industry / vulnerable
sites) in buffers around the **19 CAMS grid points**, then gave each ward the
values of its nearest grid point (build_dataset.py ~L345-357). The result:
`road_km_3km` had 19 distinct values across 289 wards, so 285 of them were
copies. Ward geometry was available the whole time.

This script recomputes every source/exposure feature on the **ward's own
geometry**, and adds the spatial graph a GNN needs.

Outputs (data/gnn/):
  nodes_static.parquet  289 wards x static features (real, per-ward)
  edges.parquet         ward adjacency: shared-border + k-nearest fallback

Run:  python -m src.data.build_ward_static
=================================================================
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from src import config as C

BASE = Path(__file__).resolve().parents[2]
GNN = BASE / "data" / "gnn"
BACKUP = BASE / "data" / "final" / "model_ready_backup.parquet"

# Metric CRS for Delhi (UTM 43N) — buffers/lengths in metres, not degrees.
CRS_M = "EPSG:32643"
ROAD_BUF_KM = 3.0
IND_BUF_KM = 5.0
VULN_BUF_KM = 3.0

# capacity weights by road class (kept identical to build_dataset.py so the
# feature keeps its original meaning — only the geometry it is measured on changes)
ROAD_W = {"motorway": 5, "trunk": 4, "primary": 3, "secondary": 2, "tertiary": 1,
          "motorway_link": 2.5, "trunk_link": 2, "primary_link": 1.5,
          "secondary_link": 1, "tertiary_link": 0.5}
DEF_LANES = {"motorway": 3, "trunk": 2, "primary": 2, "secondary": 2, "tertiary": 1}


def log(m): print(f"[ward-static] {m}", flush=True)


def load_wards():
    g = gpd.read_file(C.DIRS["wards"] / "delhi_wards.geojson")
    n0 = len(g)
    # geojson feature 192 has Ward_No=None AND Ward_Name=None -> a junk record
    # that produced 26,304 null-ward_id rows in the old dataset. Drop it.
    g = g[g["Ward_No"].notna() & g["Ward_Name"].notna()].copy()
    g["ward_id"] = g["Ward_No"].astype(str).str.strip()
    g["ward_name"] = g["Ward_Name"].astype(str).str.strip()
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    log(f"wards: {n0} features -> {len(g)} valid (dropped {n0-len(g)} null/empty)")
    assert g["ward_id"].is_unique, "ward_id must be unique"
    return g.reset_index(drop=True)


def load_roads():
    p = C.DIRS["roads"] / "osm_major_roads.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for el in data.get("elements", []):
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        tg = el.get("tags", {})
        hw = tg.get("highway", "tertiary")
        w = ROAD_W.get(hw, 1)
        try:
            lanes = float(str(tg.get("lanes", "")).split(";")[0])
        except (ValueError, TypeError):
            lanes = DEF_LANES.get(hw.replace("_link", ""), 1)
        rows.append({"cap_per_km": w * lanes,
                     "geometry": LineString([(n["lon"], n["lat"]) for n in geom])})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(CRS_M)
    log(f"roads: {len(gdf)} ways")
    return gdf


def load_points(path: Path, label: str):
    if not path.exists():
        log(f"{label}: MISSING {path.name}")
        return gpd.GeoDataFrame({"geometry": []}, crs=CRS_M)
    data = json.loads(path.read_text(encoding="utf-8"))
    pts = []
    for el in data.get("elements", []):
        if el.get("type") == "node":
            pts.append((el["lon"], el["lat"]))
        elif "center" in el:
            pts.append((el["center"]["lon"], el["center"]["lat"]))
        elif el.get("geometry"):
            g = el["geometry"][0]
            pts.append((g["lon"], g["lat"]))
    gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in pts], crs="EPSG:4326").to_crs(CRS_M)
    log(f"{label}: {len(gdf)} features")
    return gdf


def road_stats(wards_m, roads, buffer_km, suffix):
    """Length + capacity of road actually INSIDE each ward's buffer.

    Improvement over the old code: it summed a segment's FULL length whenever
    the segment midpoint fell within the radius. Here the road is clipped to
    the buffer, so a 40 km highway grazing the edge contributes only the part
    that is really there.
    """
    buf = wards_m[["ward_id", "geometry"]].copy()
    if buffer_km:
        buf["geometry"] = buf.geometry.centroid.buffer(buffer_km * 1000)
    inter = gpd.overlay(roads, buf, how="intersection", keep_geom_type=True)
    inter["km"] = inter.geometry.length / 1000.0
    inter["cap"] = inter["km"] * inter["cap_per_km"]
    agg = inter.groupby("ward_id").agg(**{
        f"road_km_{suffix}": ("km", "sum"),
        f"road_capacity_{suffix}": ("cap", "sum")}).reset_index()
    return agg


def point_counts(wards_m, pts, buffer_km, colname):
    if len(pts) == 0:
        return pd.DataFrame({"ward_id": [], colname: []})
    buf = wards_m[["ward_id", "geometry"]].copy()
    if buffer_km:
        buf["geometry"] = buf.geometry.centroid.buffer(buffer_km * 1000)
    j = gpd.sjoin(pts, buf, how="inner", predicate="within")
    return j.groupby("ward_id").size().rename(colname).reset_index()


def build_edges(wards_m, k_fallback=4, snap_m=50):
    """Ward graph: shared borders, plus k-nearest for any isolated ward.

    Edges carry distance and bearing so a GNN can gate messages on wind
    direction (pollution transport) rather than treating space as isotropic.

    NOTE: a strict `touches` predicate is unusable here — the ward polygons
    have sliver gaps, so it found only 558 pairs (mean degree 1.9) and left 52
    wards isolated. Snapping each polygon out by `snap_m` and testing
    `intersects` recovers true queen contiguity.
    """
    w = wards_m[["ward_id", "geometry"]].reset_index(drop=True)
    snapped = w.copy()
    snapped["geometry"] = snapped.geometry.buffer(snap_m)
    j = gpd.sjoin(snapped, snapped, how="inner", predicate="intersects")
    pairs = set()
    for a, b in zip(j["ward_id_left"], j["ward_id_right"]):
        if a != b:
            pairs.add((a, b))
    log(f"edges: {len(pairs)} directed border-adjacency pairs (snap {snap_m} m)")

    cent = w.copy()
    cent["geometry"] = cent.geometry.centroid
    xy = {r.ward_id: (r.geometry.x, r.geometry.y) for r in cent.itertuples()}

    # any ward with no neighbour (islands / slivers) gets k nearest centroids
    have = {a for a, _ in pairs}
    isolated = [wid for wid in w["ward_id"] if wid not in have]
    if isolated:
        ids = list(xy.keys())
        arr = np.array([xy[i] for i in ids])
        for wid in isolated:
            d = np.linalg.norm(arr - np.array(xy[wid]), axis=1)
            for idx in np.argsort(d)[1:k_fallback + 1]:
                pairs.add((wid, ids[idx]))
                pairs.add((ids[idx], wid))
        log(f"edges: {len(isolated)} isolated wards linked to {k_fallback} nearest")

    rows = []
    for a, b in sorted(pairs):
        (ax, ay), (bx, by) = xy[a], xy[b]
        dx, dy = bx - ax, by - ay
        dist_km = float(np.hypot(dx, dy) / 1000.0)
        # bearing FROM a TO b, degrees clockwise from north (matches wind convention)
        bearing = float((np.degrees(np.arctan2(dx, dy))) % 360)
        rows.append((a, b, dist_km, bearing))
    e = pd.DataFrame(rows, columns=["src_ward", "dst_ward", "dist_km", "bearing_deg"])
    log(f"edges: {len(e)} total (mean degree {len(e)/len(w):.1f})")
    return e


def main():
    GNN.mkdir(parents=True, exist_ok=True)
    wards = load_wards()
    wards_m = wards.to_crs(CRS_M)

    # centroid must be taken in the metric CRS, then projected back to lat/lon
    cent = wards_m.geometry.centroid.to_crs("EPSG:4326")
    nodes = pd.DataFrame({
        "ward_id": wards["ward_id"].values,
        "ward_name": wards["ward_name"].values,
        "ward_lat": cent.y.values,
        "ward_lon": cent.x.values,
        "ward_area_km2": (wards_m.geometry.area / 1e6).values,
    })

    roads = load_roads()
    inds = load_points(C.DIRS["industries"] / "osm_industries.json", "industries")
    vuln = load_points(C.DIRS["vulnerability"] / "osm_vulnerability.json", "vulnerability")

    log("computing per-ward road stats (3km buffer + intrinsic) ...")
    nodes = nodes.merge(road_stats(wards_m, roads, ROAD_BUF_KM, "3km"), on="ward_id", how="left")
    nodes = nodes.merge(road_stats(wards_m, roads, None, "in_ward"), on="ward_id", how="left")

    log("computing per-ward point counts ...")
    nodes = nodes.merge(point_counts(wards_m, inds, IND_BUF_KM, "industry_count_5km"), on="ward_id", how="left")
    nodes = nodes.merge(point_counts(wards_m, inds, None, "industry_count_in_ward"), on="ward_id", how="left")
    nodes = nodes.merge(point_counts(wards_m, vuln, VULN_BUF_KM, "vulnerable_sites_3km"), on="ward_id", how="left")
    nodes = nodes.merge(point_counts(wards_m, vuln, None, "vulnerable_sites_in_ward"), on="ward_id", how="left")

    for c in nodes.columns:
        if c.startswith(("road_", "industry_", "vulnerable_")):
            nodes[c] = nodes[c].fillna(0.0)

    # road density is the physically meaningful quantity for a ward
    nodes["road_km_per_km2"] = nodes["road_km_in_ward"] / nodes["ward_area_km2"].clip(lower=0.01)

    # --- carry over the genuinely ward-level layers already computed upstream
    # (population / elevation / land-use were always per-ward; only the OSM
    #  features were grid copies). Emissions come from ward_emissions.csv.
    keep = ["ward_id", "point_id", "population_sum", "population_density_mean",
            "elevation_mean", "lu_builtup_fraction", "lu_tree_fraction", "lu_majority_class"]
    bk = pd.read_parquet(BACKUP, columns=keep).drop_duplicates("ward_id")
    bk["ward_id"] = bk["ward_id"].astype(str).str.strip()
    nodes = nodes.merge(bk, on="ward_id", how="left")

    emis_p = C.PROCESSED_DIR / "ward_emissions.csv"
    if emis_p.exists():
        em = pd.read_csv(emis_p)
        em["ward_id"] = em["ward_id"].astype(str).str.strip()
        em = em.drop_duplicates("ward_id")
        nodes = nodes.merge(em.drop(columns=[c for c in ("ward_name",) if c in em.columns]),
                            on="ward_id", how="left")
        log(f"merged {em.shape[1]-1} emission columns")

    nodes = nodes.sort_values("ward_id").reset_index(drop=True)
    nodes["node_idx"] = np.arange(len(nodes))  # stable GNN node index

    edges = build_edges(wards_m)
    idx = dict(zip(nodes["ward_id"], nodes["node_idx"]))
    edges = edges[edges["src_ward"].isin(idx) & edges["dst_ward"].isin(idx)].copy()
    edges["src"] = edges["src_ward"].map(idx)
    edges["dst"] = edges["dst_ward"].map(idx)

    nodes.to_parquet(GNN / "nodes_static.parquet", index=False)
    edges.to_parquet(GNN / "edges.parquet", index=False)
    log(f"SAVED nodes_static.parquet {nodes.shape}  edges.parquet {edges.shape}")

    # --- prove the fix: these used to have 16-19 distinct values across 289 wards
    log("distinct values across wards (was 19/16/17 when copied from grid points):")
    for c in ["road_km_3km", "road_capacity_3km", "industry_count_5km",
              "vulnerable_sites_3km", "road_km_in_ward", "road_km_per_km2"]:
        log(f"  {c:26s} {nodes[c].nunique():4d} distinct")


if __name__ == "__main__":
    main()
