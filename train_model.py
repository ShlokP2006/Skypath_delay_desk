"""Train the delay model and write everything the app needs to artifacts/.

Real data (put train.csv and test.csv in data/):

    python train_model.py --sample-frac 0.4

Quick smoke test on synthetic data (numbers are meaningless):

    python train_model.py --demo
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib

from src import config, data, lookups, modeling


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default=str(config.DATA_DIR / "train.csv"))
    p.add_argument("--test", default=str(config.DATA_DIR / "test.csv"))
    p.add_argument(
        "--sample-frac",
        type=float,
        default=0.4,
        help="Random fraction of rows to keep from each file (1.0 = everything). Default 0.4.",
    )
    p.add_argument("--trees", type=int, default=1500, help="Maximum boosting rounds (early stopping applies).")
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demo", action="store_true", help="Use synthetic demo data instead of the real CSVs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = "real"
    frac = args.sample_frac

    if args.demo:
        import make_demo_data

        source = "demo"
        frac = 1.0
        args.train = str(config.DATA_DIR / "demo_train.csv")
        args.test = str(config.DATA_DIR / "demo_test.csv")
        if not (config.DATA_DIR / "demo_train.csv").exists():
            make_demo_data.main()

    print(f"[1/5] Reading training data ({source}, sample fraction {frac})")
    train = data.read_flights(args.train, sample_frac=frac, seed=args.seed)
    print(f"      {len(train):,} training rows, delay rate {train[config.TARGET].mean():.1%}")

    print("[2/5] Building lookup tables and chart summaries")
    lk = lookups.build_lookups(train)
    eda = lookups.build_eda(train)

    print("[3/5] Fitting LightGBM")
    model, threshold = modeling.fit_model(
        train, n_estimators=args.trees, learning_rate=args.learning_rate, seed=args.seed
    )

    bundle = {
        "model": model,
        "features": config.FEATURES,
        "categorical": config.CATEGORICAL,
        "categories": {c: [str(v) for v in train[c].cat.categories] for c in config.CATEGORICAL},
        "threshold": threshold,
    }

    print("[4/5] Evaluating on the test file")
    try:
        test = data.read_flights(args.test, sample_frac=frac, seed=args.seed + 1)
    except FileNotFoundError:
        print("      test file not found, holding out 15% of training rows instead")
        test = train.sample(frac=0.15, random_state=args.seed)
    metrics = modeling.evaluate(
        model, modeling.align_frame(test, bundle), test[config.TARGET], threshold
    )
    metrics.update(
        source=source,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_train=int(len(train)),
        sample_frac=frac,
        base_rate_train=float(train[config.TARGET].mean()),
        best_iteration=int(model.best_iteration_ or args.trees),
        importance=modeling.feature_importance(model, config.FEATURES),
    )

    print("[5/5] Saving artifacts")
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH, compress=3)
    config.LOOKUPS_PATH.write_text(json.dumps(lk))
    config.EDA_PATH.write_text(json.dumps(eda))
    config.METRICS_PATH.write_text(json.dumps(lookups.jsonable(metrics), indent=2))

    print()
    print(f"  ROC-AUC {metrics['roc_auc']:.3f}   PR-AUC {metrics['pr_auc']:.3f}   "
          f"top-decile lift {metrics['top_decile_lift']:.2f}x")
    print(f"  At threshold {threshold:.2f}: precision {metrics['precision']:.2f}, recall {metrics['recall']:.2f}")
    print(f"  Artifacts written to {config.ARTIFACT_DIR}. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
