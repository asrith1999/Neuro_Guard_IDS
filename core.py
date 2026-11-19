"""Shared schema and prediction helpers for a local educational IDS demo."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "duration_ms", "dst_port", "src_bytes", "dst_bytes", "packets",
    "syn_count", "failed_logins", "connections_per_minute",
]
CATEGORICAL_FEATURES = ["protocol"]
FEATURES = ["protocol", *NUMERIC_FEATURES]
LABELS = {0: "Normal", 1: "Suspicious"}


def check_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return validated features in a consistent order; never run network requests."""
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError("Missing CSV column(s): " + ", ".join(missing))
    clean = frame.loc[:, FEATURES].copy()
    clean["protocol"] = clean["protocol"].astype("string").str.strip().str.upper()
    if clean["protocol"].isna().any() or (~clean["protocol"].isin(["TCP", "UDP", "ICMP"])).any():
        raise ValueError("protocol must be TCP, UDP, or ICMP in every row.")
    for name in NUMERIC_FEATURES:
        clean[name] = pd.to_numeric(clean[name], errors="coerce")
        if not np.isfinite(clean[name].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} must contain a valid number in every row.")
        if (clean[name] < 0).any():
            raise ValueError(f"{name} must not contain negative values.")
    if (clean["dst_port"] > 65535).any():
        raise ValueError("dst_port must be between 0 and 65535.")
    return clean


def check_labels(frame: pd.DataFrame) -> np.ndarray:
    if "label" not in frame.columns:
        raise ValueError("Training data needs a 'label' column: 0=Normal, 1=Suspicious.")
    labels = (
        frame["label"].astype("string").str.strip().str.lower().map(
            {"0": 0, "normal": 0, "benign": 0,
             "1": 1, "suspicious": 1, "attack": 1, "malicious": 1}
        )
    )
    if labels.isna().any() or labels.nunique() != 2:
        raise ValueError("label must have both classes, using 0/1 or Normal/Suspicious.")
    if labels.value_counts().min() < 10:
        raise ValueError("Training needs at least 10 rows of each class.")
    return labels.to_numpy(dtype=int)


def make_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("protocol", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    classifier = MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
        max_iter=350, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def predict_frame(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    features = check_features(frame)
    # Explicitly locate the positive class rather than relying on implicit order.
    class_index = list(model.classes_).index(1)
    scores = model.predict_proba(features)[:, class_index]
    result = frame.copy()
    result["prediction"] = np.where(scores >= 0.5, "Suspicious", "Normal")
    result["suspicious_score"] = np.round(scores, 4)
    return result
