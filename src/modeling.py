"""Model training, evaluation and inference helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------
# Risk categories (VFR / MVFR / IFR / LIFR)
# --------------------------------------------------------------------------


def category_bounds(base_rate: float) -> list[float]:
    """Probability cut-offs between the four categories.

    They scale with the average delay rate so "low risk" always means clearly
    better than the average departure, whatever the dataset's base rate is.
    """
    return [min(0.8 * base_rate, 0.90), min(1.5 * base_rate, 0.95), min(2.5 * base_rate, 0.99)]


def flight_category(p: float, base_rate: float) -> dict:
    index = sum(p >= bound for bound in category_bounds(base_rate))
    return {**config.CATEGORIES[index], "index": index}


# --------------------------------------------------------------------------
# Building model input
# --------------------------------------------------------------------------


def _get(d: dict, *keys, default=np.nan):
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return default if d is None else d


def build_row(inputs: dict, lookups: dict) -> dict:
    """Combine what the person typed in with typical values from the lookups."""
    airport = inputs["DEPARTING_AIRPORT"]
    carrier = inputs["CARRIER_NAME"]
    ap = lookups["airports"].get(airport, {})
    cr = lookups["carriers"].get(carrier, {})

    pair = lookups["pairs"].get(f"{carrier}||{airport}")
    if pair is None:
        pair = np.nan

    row = dict(inputs)
    row.update(
        LATITUDE=_get(ap, "latitude"),
        LONGITUDE=_get(ap, "longitude"),
        AIRPORT_FLIGHTS_MONTH=_get(ap, "airport_flights_month"),
        AVG_MONTHLY_PASS_AIRPORT=_get(ap, "avg_monthly_pass"),
        AIRLINE_FLIGHTS_MONTH=_get(cr, "airline_flights_month"),
        AVG_MONTHLY_PASS_AIRLINE=_get(cr, "avg_monthly_pass"),
        FLT_ATTENDANTS_PER_PASS=_get(cr, "flt_attendants_per_pass"),
        GROUND_SERV_PER_PASS=_get(cr, "ground_serv_per_pass"),
        AIRLINE_AIRPORT_FLIGHTS_MONTH=pair,
    )
    return row


def make_frame(rows: list[dict], bundle: dict) -> pd.DataFrame:
    """Turn input rows into a frame with the exact dtypes the model was trained on."""
    df = pd.DataFrame(rows)[bundle["features"]].copy()
    for col in bundle["categorical"]:
        df[col] = pd.Categorical(df[col].astype(object), categories=bundle["categories"][col])
    for col in bundle["features"]:
        if col not in bundle["categorical"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    return df


def align_frame(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Give a test DataFrame the same category levels as the training data."""
    X = df[bundle["features"]].copy()
    for col in bundle["categorical"]:
        X[col] = pd.Categorical(X[col].astype(object), categories=bundle["categories"][col])
    return X


def predict(bundle: dict, rows: list[dict]) -> np.ndarray:
    return bundle["model"].predict_proba(make_frame(rows, bundle))[:, 1]


def contributions(bundle: dict, row: dict) -> pd.Series:
    """Per-feature push on the log-odds of a delay (LightGBM's built-in SHAP values)."""
    X = make_frame([row], bundle)
    values = bundle["model"].predict(X, pred_contrib=True)[0]
    return pd.Series(values[:-1], index=bundle["features"])


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def fit_model(train_df: pd.DataFrame, n_estimators=1500, learning_rate=0.05, seed=42):
    import warnings

    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.model_selection import train_test_split

    # LightGBM 4.7 deprecates eval_set in favour of eval_X / eval_y, but older
    # 4.x releases only know eval_set, so keep it and hide the warning.
    warnings.filterwarnings("ignore", message=".*eval_set.*")

    X = train_df[config.FEATURES]
    y = train_df[config.TARGET]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=127,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(100)],
    )

    # Decision threshold that maximises F1 on the held-out validation slice.
    from sklearn.metrics import precision_recall_curve

    p_va = model.predict_proba(X_va)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_va, p_va)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    threshold = float(thresholds[int(np.argmax(f1))])
    return model, threshold


def evaluate(model, X: pd.DataFrame, y, threshold: float) -> dict:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    y = np.asarray(y)
    p = model.predict_proba(X)[:, 1]
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    fpr, tpr, _ = roc_curve(y, p)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 150).astype(int))

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")

    top = np.argsort(-p)[: max(1, len(p) // 10)]
    top_decile_rate = float(y[top].mean())

    return {
        "n_test": int(len(y)),
        "base_rate_test": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "top_decile_delay_rate": top_decile_rate,
        "top_decile_lift": top_decile_rate / float(y.mean()),
        "roc": {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()},
        "calibration": {"predicted": mean_pred.tolist(), "observed": frac_pos.tolist()},
    }


def feature_importance(model, features: list[str]) -> dict:
    gain = model.booster_.feature_importance(importance_type="gain").astype(float)
    gain = gain / gain.sum()
    return dict(sorted(zip(features, gain.tolist()), key=lambda kv: -kv[1]))
