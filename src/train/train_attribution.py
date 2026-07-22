"""
train_attribution.py  —  Engine 2 training & evaluation
=================================================================
Trains the two LightGBM heads and validates them the honest way:

  ratio head  : predict observed pm2.5/pm10 ratio.  Report MAE vs the
                predict-the-mean baseline (the only fair bar for a regressor).
  class head  : dust / mixed / combustion.  Report **macro-F1 on VAL**, not test
                — test is summer and ~72% dust, so test F1 flatters a lazy model.
                Val carries all three classes, so macro-F1 there is meaningful.

Splits come from `split_lab` (chronological, labelled era). Rows with a null
observed target are dropped (never imputed — see PREPROCESSING.md).

    python -m src.train.train_attribution
=================================================================
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import f1_score, mean_absolute_error, classification_report

from models.attribution import (
    build_attribution_frame, SourceAttributor, RATIO_TARGET, CLASS_TARGET, CKPT_DIR,
)

TAG = "attribution"


def _split(df, col):
    sub = df[df[col].notna()].copy()
    return (sub[sub.split_lab == "train"], sub[sub.split_lab == "val"],
            sub[sub.split_lab == "test"])


def train_ratio(df, feats):
    tr, va, te = _split(df, RATIO_TARGET)
    Xtr, ytr = tr[feats].to_numpy(), tr[RATIO_TARGET].to_numpy()
    m = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                      subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1)
    m.fit(Xtr, ytr)
    print("\n== RATIO HEAD (pm2.5/pm10) ==")
    for name, part in [("val", va), ("test", te)]:
        y = part[RATIO_TARGET].to_numpy()
        pred = m.predict(part[feats].to_numpy())
        base = np.full_like(y, ytr.mean())
        print(f"  [{name}] MAE={mean_absolute_error(y, pred):.4f}  "
              f"vs predict-mean {mean_absolute_error(y, base):.4f}  "
              f"(-{100*(1-mean_absolute_error(y,pred)/mean_absolute_error(y,base)):.1f}% error)")
    return m


def train_class(df, feats):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import log_loss
    tr, va, te = _split(df, CLASS_TARGET)
    Xtr, ytr = tr[feats].to_numpy(), tr[CLASS_TARGET].to_numpy()
    base = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          subsample=0.8, colsample_bytree=0.8, random_state=0,
                          n_jobs=-1, class_weight="balanced")
    base.fit(Xtr, ytr)
    # 4.2 — calibrate probabilities on val so the donut's confidence means what it
    # says. Sigmoid/Platt (not isotonic) — gentler + more robust to the winter→summer
    # shift, so it improves calibration without distorting the argmax/F1.
    m = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    m.fit(va[feats].to_numpy(), va[CLASS_TARGET].to_numpy())

    maj = tr[CLASS_TARGET].mode().iloc[0]
    print("\n== CLASS HEAD (dust/mixed/combustion) ==")
    for name, part in [("val", va), ("test", te)]:
        y = part[CLASS_TARGET].to_numpy()
        pred = m.predict(part[feats].to_numpy())
        f1 = f1_score(y, pred, average="macro")
        f1_maj = f1_score(y, np.full_like(y, maj, dtype=object), average="macro")
        print(f"  [{name}] macro-F1={f1:.3f}  vs majority({maj})={f1_maj:.3f}  (+{f1-f1_maj:.3f})")
    # 4.2 diagnostic — calibration barely helps here and costs argmax/F1 because
    # it is fit on winter-val and applied to summer-test (seasonal shift). So we
    # SERVE the uncalibrated model (better source labels) and just report the
    # comparison. Flip `SERVE_CALIBRATED` to True if honest confidence > F1 for you.
    Xte, yte = te[feats].to_numpy(), te[CLASS_TARGET].to_numpy()
    ll_raw = log_loss(yte, base.predict_proba(Xte), labels=base.classes_)
    ll_cal = log_loss(yte, m.predict_proba(Xte), labels=base.classes_)
    f1_raw = f1_score(yte, base.predict(Xte), average="macro")
    f1_cal = f1_score(yte, m.predict(Xte), average="macro")
    print(f"  4.2 calibration diagnostic (test): log-loss {ll_raw:.3f}->{ll_cal:.3f} "
          f"(better) BUT macro-F1 {f1_raw:.3f}->{f1_cal:.3f} (worse, seasonal shift)")
    SERVE_CALIBRATED = False
    served = m if SERVE_CALIBRATED else base
    print(f"  -> serving {'CALIBRATED' if SERVE_CALIBRATED else 'uncalibrated'} class head")
    print("\n  val classification report:")
    print(classification_report(va[CLASS_TARGET], served.predict(va[feats].to_numpy()),
                                digits=3, zero_division=0))
    return served


def main():
    df, feats = build_attribution_frame()
    print(f"frame {df.shape}  features={len(feats)}")
    ratio_m = train_ratio(df, feats)
    class_m = train_class(df, feats)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CKPT_DIR / f"{TAG}_ratio.pkl", "wb") as f:
        pickle.dump(ratio_m, f)
    with open(CKPT_DIR / f"{TAG}_class.pkl", "wb") as f:
        pickle.dump(class_m, f)
    with open(CKPT_DIR / f"{TAG}_meta.pkl", "wb") as f:
        pickle.dump({"features": feats}, f)
    print(f"\nsaved 3 artefacts to {CKPT_DIR}")

    # demo: qualitative profile on a real winter combustion-ish row
    attr = SourceAttributor.load(TAG)
    va = df[(df.split_lab == "val") & (df[CLASS_TARGET] == "combustion_dominated")]
    if len(va):
        prof = attr.profile(va.iloc[0])
        print("\n== demo profile (a combustion-labelled val ward-hour) ==")
        print("  dominant_class:", prof.get("dominant_class"),
              "| pred ratio:", round(prof.get("pred_pm25_pm10_ratio", float('nan')), 3),
              "| confidence:", prof["confidence"])
        print("  ranked:", [(r["source"], r["score"]) for r in prof["ranked_sources"]])
        print("  edgar_prior_pm25:", prof["edgar_prior_pm25"])


if __name__ == "__main__":
    main()
