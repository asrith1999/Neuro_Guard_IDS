"""Train and evaluate a tiny ANN on labeled flow summary CSV data."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

from core import FEATURES, check_features, check_labels, make_pipeline

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "synthetic_training.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


def train(data_path: Path) -> dict:
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Run generate_data.py first.")
    dataset = pd.read_csv(data_path)
    X = check_features(dataset)
    y = check_labels(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    model = make_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "data_source": data_path.name,
        "dataset_rows": int(len(dataset)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "9 original input fields -> preprocessor -> 32 -> 16 -> binary output",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Load this file only if you created it yourself; joblib/pickle files can execute code.
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nModel trained successfully!")
    print(f"Training rows: {metrics['train_rows']}; testing rows: {metrics['test_rows']}")
    print(f"Accuracy:  {metrics['accuracy']:.1%}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall:    {metrics['recall']:.1%}")
    print(f"F1 score:  {metrics['f1']:.1%}")
    print(f"Saved: {MODEL_PATH}")
    print("NOTE: Accuracy on simulated data is not a measure of real IDS performance.")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuroGuard educational neural-network IDS")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="CSV file with schema from README, including a label column")
    args = parser.parse_args()
    train(args.data)
