"""Classify a CSV file using the previously trained local model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from core import predict_frame
from train_model import MODEL_PATH

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify existing flow summary CSV")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "sample_flows.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "predictions.csv")
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        parser.error("Model missing: run python train_model.py first")
    # Only load a model that you generated locally, never an untrusted joblib file.
    model = joblib.load(MODEL_PATH)["model"]
    flows = pd.read_csv(args.input)
    if flows.empty:
        parser.error("Input CSV is empty")
    results = predict_frame(model, flows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results[["protocol", "dst_port", "prediction", "suspicious_score"]].head(12).to_string(index=False))
    print(f"\nSaved {len(results)} classifications to {args.output}")
    print("Scores are NOT validated real-world threat probabilities.")


if __name__ == "__main__":
    main()
