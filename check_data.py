"""
check_data.py  — one-shot health check for model_ready.parquet
Run:  python check_data.py
Prints PASS/WARN for every check so you can eyeball the dataset in ~10s.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
F = BASE / "data" / "final" / "model_ready.parquet"
GT = BASE / "data" / "processed" / "cpcb_ground_truth.csv"


def line(ok, msg): print(f"  [{'PASS' if ok else 'WARN'}] {msg}")


def main():
    df = pd.read_parquet(F)
    print(f"\n=== FILE ===  {F.name}")
    print(f"  {df.shape[0]:,} rows x {df.shape[1]} cols | "
          f"{df['ward_id'].nunique()} wards | "
          f"{pd.to_datetime(df['time']).min().date()} -> {pd.to_datetime(df['time']).max().date()}")

    print("\n=== 1. SEASONALITY (winter must beat summer) ===")
    mm = df.groupby("month")["aqi"].mean().round(0)
    print("  monthly mean AQI:", mm.to_dict())
    w, s = mm.reindex([11, 12, 1]).mean(), mm.reindex([4, 5, 6]).mean()
    line(w > s, f"winter(Nov-Jan)={w:.0f} > summer(Apr-Jun)={s:.0f}")

    print("\n=== 2. WARD DISTINCTNESS (downscaling worked) ===")
    t0 = df["time"].iloc[len(df) // 2]
    u = df[df["time"] == t0]["aqi"].nunique()
    line(u > 20, f"{u} distinct AQI values across wards at one hour (was 4 before)")

    print("\n=== 3. NULLS in key columns ===")
    for c in ["aqi", "target_aqi_t24", "road_capacity_3km", "elevation_mean",
              "ward_hist_aqi", "industry_upwind", "emis_total_pm25"]:
        pct = 100 * df[c].isna().mean()
        line(pct < 1, f"{c}: {pct:.2f}% null")

    print("\n=== 4. TARGET ALIGNMENT (no leakage) ===")
    wid = df["ward_id"].dropna().iloc[0]
    g = df[df["ward_id"] == wid].sort_values("time").reset_index()
    ok = np.isclose(g["aqi"].iloc[24], g["target_aqi_t24"].iloc[0], equal_nan=True)
    line(ok, f"target_aqi_t24[0]={g['target_aqi_t24'].iloc[0]:.0f} == aqi[+24h]={g['aqi'].iloc[24]:.0f}")

    print("\n=== 5. FORECAST BASELINE TO BEAT (test split) ===")
    for h in (24, 48, 72):
        tgt = f"target_aqi_t{h}"
        t = df[df.split == "test"].dropna(subset=[tgt, "persistence_aqi_t24"])
        rmse = np.sqrt(((t[tgt] - t["persistence_aqi_t24"]) ** 2).mean())
        mae = (t[tgt] - t["persistence_aqi_t24"]).abs().mean()
        print(f"  t+{h}h persistence  RMSE={rmse:.1f}  MAE={mae:.1f}")

    print("\n=== 6. EDGAR EMISSIONS present & varied ===")
    em = [c for c in df.columns if c.startswith("emis_")]
    nuniq = df.groupby("ward_id")["emis_total_pm25"].first().nunique()
    line(len(em) == 20 and nuniq > 5, f"{len(em)} emis cols, {nuniq} distinct ward values")

    print("\n=== 7. NO LEAKING COLUMNS in feature list ===")
    try:
        import feature_lists as fl
        leak = [c for c in fl.forecast_features if c in fl.EXCLUDE_ALWAYS]
        miss = [c for c in fl.forecast_features if c not in df.columns]
        line(not leak and not miss, f"forecast: {len(fl.forecast_features)} feats, leak={leak}, missing={miss}")
    except Exception as e:
        line(False, f"feature_lists check skipped: {e}")

    print("\n=== 8. CPCB GROUND TRUTH (validation set) ===")
    if GT.exists():
        gt = pd.read_csv(GT)
        line(gt["cpcb_pm25"].notna().sum() > 1000,
             f"{gt['cpcb_pm25'].notna().sum()} PM2.5 readings, {gt['station_id'].nunique()} stations")
    print("\nDone. Any [WARN] above is worth a look.\n")


if __name__ == "__main__":
    main()
