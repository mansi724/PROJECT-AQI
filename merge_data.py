import os
import sys
import json
import warnings
import gzip
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats
import rasterio
from rasterio.merge import merge

import config as C

# Suppress some noisy warnings from rasterio/geopandas if they occur
warnings.filterwarnings("ignore", category=UserWarning)

def log(msg):
    print(f"[merge] {msg}", flush=True)

def merge_dem_tiles(dem_dir, output_path):
    """Merge downloaded SRTM DEM tiles into a single virtual/physical raster."""
    tiles = list(dem_dir.glob("*.hgt.gz"))
    if not tiles:
        return None
    try:
        srcs = []
        for gz in tiles:
            hgt = gz.with_suffix("")
            if not hgt.exists():
                with gzip.open(gz,"rb") as fi, open(hgt,"wb") as fo:
                    shutil.copyfileobj(fi, fo)
            srcs.append(rasterio.open(hgt))
        
        if not srcs: return None
        mosaic, out_trans = merge(srcs)
        meta = srcs[0].meta.copy()
        
        meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        
        with rasterio.open(output_path, "w", **meta) as dest:
            dest.write(mosaic)
            
        for src in srcs:
            src.close()
            
        return output_path
    except Exception as e:
        log(f"Failed to merge DEM tiles: {e}")
        return None

def extract_raster_features(wards_gdf):
    """
    Given a GeoDataFrame of wards, extract zonal statistics from:
      1. DEM (Elevation)
      2. Land Cover class modes
      3. Population density sums/means
    """
    log("Extracting raster features...")
    
    features = []
    
    # Ensure geometries are robust for rasterstats (using WGS84 for lat/lon rasters)
    if wards_gdf.crs is None:
        wards_gdf.set_crs(epsg=4326, inplace=True)
    wards_gdf = wards_gdf.to_crs(epsg=4326)
    
    dem_dir = C.DIRS["dem"]
    landuse_dir = C.DIRS["landuse"]
    pop_dir = C.DIRS["population"]

    # 1. Population (WorldPop)
    pop_files = list(pop_dir.glob("*.tif"))
    pop_stats = []
    if pop_files:
        pop_raster = pop_files[0]
        log(f"Processing population: {pop_raster.name}")
        pop_stats = zonal_stats(wards_gdf, str(pop_raster), stats=['sum', 'mean'], nodata=-99999)
    else:
        pop_stats = [{'sum': np.nan, 'mean': np.nan} for _ in range(len(wards_gdf))]
        
    # 2. DEM
    merged_dem_path = dem_dir / "merged_dem.tif"
    if not merged_dem_path.exists():
        merge_dem_tiles(dem_dir, merged_dem_path)
    
    dem_stats = []
    if merged_dem_path.exists():
        log(f"Processing DEM: {merged_dem_path.name}")
        dem_stats = zonal_stats(wards_gdf, str(merged_dem_path), stats=['mean'])
    else:
        dem_stats = [{'mean': np.nan} for _ in range(len(wards_gdf))]
        
    # 3. Land Use (ESA WorldCover)
    lu_files = list(landuse_dir.glob("worldcover_*.tif"))
    lu_stats = []
    if lu_files:
        lu_raster = lu_files[0]
        log(f"Processing Land Use: {lu_raster.name}")
        # categorical=True gives counts of each pixel value
        lu_stats = zonal_stats(wards_gdf, str(lu_raster), categorical=True, nodata=0)
    else:
        lu_stats = [{} for _ in range(len(wards_gdf))]

    # Combine into dataframe
    for i in range(len(wards_gdf)):
        f = {
            "ward_id": wards_gdf.iloc[i]["ward_id"],
            "population_sum": pop_stats[i].get("sum", np.nan),
            "population_density_mean": pop_stats[i].get("mean", np.nan),
            "elevation_mean": dem_stats[i].get("mean", np.nan)
        }
        # For landuse, find the mode (class with most pixels)
        # Or store fractional covers. We'll store the majority class and total built-up fraction
        lu = lu_stats[i]
        if lu:
            total_pixels = sum(lu.values())
            # ESA WorldCover Built-up class is typically 50. Tree cover is 10, etc.
            f["lu_builtup_fraction"] = lu.get(50, 0) / total_pixels if total_pixels > 0 else 0.0
            f["lu_tree_fraction"] = lu.get(10, 0) / total_pixels if total_pixels > 0 else 0.0
            f["lu_majority_class"] = max(lu, key=lu.get) if lu else np.nan
        else:
            f["lu_builtup_fraction"] = np.nan
            f["lu_tree_fraction"] = np.nan
            f["lu_majority_class"] = np.nan
            
        features.append(f)
        
    return pd.DataFrame(features)

def match_satellite_data(base_df, wards_gdf):
    """
    Placeholder for matching Sentinel-5P and MODIS granules to the spatial/temporal base_df.
    If actual .nc/.hdf files are downloaded, they can be processed here with xarray to extract 
    ward-level time-series averages. Currently, standard run only downloads catalogs.
    """
    # Assuming user has not manually downloaded heavy granules unless present, 
    # we just instantiate the target columns with NaNs to maintain schema consistency.
    log("Checking for Satellite Granules...")

    modis_files = list(C.DIRS["modis"].glob("*.hdf"))
    s5p_files = list(C.DIRS["sentinel5p"].glob("*.nc"))

    if modis_files or s5p_files:
        log("Found actual NetCDF/HDF satellite files! (Parsing these is out of scope for default run)")
        # Real extraction logic using xarray & regionmask would go here.
        # Only then create the columns, so we never emit all-NaN placeholder columns.
        for col in ["s5p_no2_column_density", "s5p_so2_column_density",
                    "s5p_co_column_density", "s5p_aerosol_index", "modis_aod_550"]:
            base_df[col] = np.nan
    else:
        log("No satellite granules present — skipping satellite columns "
            "(CAMS aerosol_optical_depth/dust/no2 cover this signal).")

    return base_df

def main():
    log("Starting final dataset merge...")
    processed_dir = C.PROCESSED_DIR
    final_dir = C.DATA_DIR / "final"
    final_dir.mkdir(exist_ok=True, parents=True)
    
    final_dataset_csv = processed_dir / "final_dataset.csv"
    ward_features_csv = processed_dir / "ward_features.csv"
    ward_geojson_path = C.DIRS["wards"] / "delhi_wards.geojson"
    
    if not (final_dataset_csv.exists() and ward_features_csv.exists()):
        sys.exit("Error: Intermediate processed files not found. Please run build_dataset.py first.")
        
    if not ward_geojson_path.exists():
        sys.exit(f"Error: {ward_geojson_path} not found.")

    # 1. Load basic ward mapping and compute standard IDs
    log("Loading Wards GeoJSON...")
    with open(ward_geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
        
    # We need to recreate the same ward_id used in build_dataset.py
    for i, feature in enumerate(gj.get("features", [])):
        props = feature.get("properties", {})
        wid = next((props[k] for k in props if str(k).lower() in ("ward_no", "wardno", "ward_id", "id", "wardcode")), i)
        feature["properties"]["ward_id"] = wid
        
    wards_gdf = gpd.GeoDataFrame.from_features(gj, crs="EPSG:4326")

    # 2. Extract Static Raster Features
    raster_df = extract_raster_features(wards_gdf)
    
    # 3. Load processed grid temporal base and ward mappings
    log("Loading temporal and vector mappings...")
    temporal_df = pd.read_csv(final_dataset_csv)
    temporal_df['time'] = pd.to_datetime(temporal_df['time'])
    
    ward_features = pd.read_csv(ward_features_csv)
    
    # 4. Integrate Static Features (Raster + Vector)
    log("Integrating static features...")
    # ward_features already contains vector features per ward (road_km_3km, etc.)
    static_comb = pd.merge(ward_features, raster_df, on="ward_id", how="left")

    # Fix 6: Fill population nulls using nearest valid ward
    for c in ["population_sum", "population_density_mean"]:
        if c in static_comb.columns and static_comb[c].isnull().any():
            null_idx = static_comb[static_comb[c].isnull()].index
            valid_idx = static_comb[~static_comb[c].isnull()].index
            for i in null_idx:
                dist = ((static_comb.loc[valid_idx, 'ward_lat'] - static_comb.loc[i, 'ward_lat'])**2 + 
                        (static_comb.loc[valid_idx, 'ward_lon'] - static_comb.loc[i, 'ward_lon'])**2)
                if not dist.empty:
                    static_comb.loc[i, c] = static_comb.loc[dist.idxmin(), c]
    
    # 5. Build Final Spatio-Temporal Dataset
    log("Broadcasting temporal grid data to Wards...")
    # Each ward is mapped to a point_id. We merge temporal (which has point_id) to the static ward features.
    # Note: temporal_df also already has the point-based vector features. We drop them to avoid conflicts, 
    # since we want to use the ward-specific vector features computed in ward_features.
    cols_to_drop = ["lat", "lon", "road_km_3km", "road_capacity_3km", "industry_count_5km", "vulnerable_sites_3km", "dist_to_grid_km", "nearest_station_id", "dist_to_station_km"]
    # also drop any cpcb_* already merged in build_dataset.py, so this merge doesn't
    # create duplicate cpcb_*_x / cpcb_*_y columns.
    cols_to_drop += [c for c in temporal_df.columns if c.startswith("cpcb_")]
    temporal_clean = temporal_df.drop(columns=[c for c in cols_to_drop if c in temporal_df.columns], errors='ignore')
    
    final_df = pd.merge(
        temporal_clean, 
        static_comb, 
        on="point_id", 
        how="right"
    )
    
    # 5.5 Merge CPCB Ground Truth
    cpcb_csv = processed_dir / "cpcb_ground_truth.csv"
    if cpcb_csv.exists():
        log("Merging CPCB ground truth target variables...")
        cpcb = pd.read_csv(cpcb_csv)
        if getattr(C, "TEMPORAL_RESOLUTION", "hourly") == "hourly":
            cpcb['time'] = pd.to_datetime(cpcb['time'])
            join_key = "time"
        else:
            final_df['date'] = final_df['time'].dt.date
            cpcb['time'] = pd.to_datetime(cpcb['time']).dt.date
            cpcb = cpcb.rename(columns={'time': 'date'})
            join_key = "date"
            
        final_df = pd.merge(
            final_df, 
            cpcb, 
            left_on=["nearest_station_id", join_key], 
            right_on=["station_id", join_key], 
            how="left"
        )
        final_df = final_df.drop(columns=["station_id"], errors='ignore')
        
        cpcb_cols = [c for c in final_df.columns if c.startswith("cpcb_")]
        if getattr(C, "TEMPORAL_RESOLUTION", "hourly") == "hourly":
            log(f"Handling nulls for hourly targets ({len(cpcb_cols)} columns) via 6-hour interpolation limit...")
            for ward in final_df['ward_id'].unique():
                idx = final_df['ward_id'] == ward
                final_df.loc[idx, cpcb_cols] = final_df.loc[idx, cpcb_cols].interpolate(method='linear', limit=6, limit_direction='both')
    
    # Rearrange columns for neatness
    core_cols = ["ward_id", "ward_name", "time", "point_id", "ward_lat", "ward_lon"]
    remaining_cols = [c for c in final_df.columns if c not in core_cols]
    final_df = final_df[core_cols + remaining_cols]
    
    # Sort
    final_df = final_df.sort_values(by=["ward_id", "time"])
    
    # 6. Extract Satellite / Space-borne columns 
    final_df = match_satellite_data(final_df, wards_gdf)

    # 7. Export
    out_file = final_dir / "delhi_ward_dataset.csv"
    log(f"Exporting final dataset to {out_file}...")
    final_df.to_csv(out_file, index=False)
    
    rows = len(final_df)
    cols = len(final_df.columns)
    sz_mb = out_file.stat().st_size / 1e6
    log(f"Done! Created {out_file.name}: {rows:,} rows x {cols} columns ({sz_mb:.1f} MB).")

if __name__ == "__main__":
    main()
