"""Generate *fictional* network-flow summary data for learning ML basics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"


def normal_flow(rng: np.random.Generator) -> dict:
    protocol = str(rng.choice(["TCP", "UDP", "ICMP"], p=[0.7, 0.25, 0.05]))
    port = int(rng.choice([80, 443, 53, 22, 123, 8080, 3389],
                          p=[0.14, 0.40, 0.18, 0.06, 0.10, 0.09, 0.03]))
    if protocol == "ICMP":
        port = 0
    return {
        "protocol": protocol,
        "duration_ms": int(np.clip(rng.lognormal(7.0, 0.9), 20, 80000)),
        "dst_port": port,
        "src_bytes": int(np.clip(rng.lognormal(7.0, 0.9), 25, 180000)),
        "dst_bytes": int(np.clip(rng.lognormal(7.5, 1.0), 20, 220000)),
        "packets": int(rng.integers(6, 45)),
        "syn_count": int(rng.integers(0, 3)) if protocol == "TCP" else 0,
        "failed_logins": int(rng.choice([0, 1], p=[0.97, 0.03])),
        "connections_per_minute": int(rng.integers(1, 28)),
        "label": 0,
    }


def suspicious_flow(rng: np.random.Generator) -> dict:
    """Three illustrative anomalous patterns, NOT real captured attacks."""
    pattern = str(rng.choice(["many_connections", "login_failures", "large_packet_burst"]))
    if pattern == "many_connections":
        return {
            "protocol": "TCP", "duration_ms": int(rng.integers(20, 500)),
            "dst_port": int(rng.integers(1, 65536)),
            "src_bytes": int(rng.integers(20, 350)),
            "dst_bytes": int(rng.integers(0, 200)),
            "packets": int(rng.integers(2, 15)),
            "syn_count": int(rng.integers(3, 16)),
            "failed_logins": 0,
            "connections_per_minute": int(rng.integers(80, 350)),
            "label": 1,
        }
    if pattern == "login_failures":
        return {
            "protocol": "TCP", "duration_ms": int(rng.integers(1500, 20000)),
            "dst_port": int(rng.choice([22, 443, 3389])),
            "src_bytes": int(rng.integers(250, 4000)),
            "dst_bytes": int(rng.integers(60, 4000)),
            "packets": int(rng.integers(15, 70)),
            "syn_count": int(rng.integers(0, 5)),
            "failed_logins": int(rng.integers(5, 22)),
            "connections_per_minute": int(rng.integers(15, 110)),
            "label": 1,
        }
    proto = str(rng.choice(["TCP", "UDP", "ICMP"]))
    return {
        "protocol": proto, "duration_ms": int(rng.integers(70, 2200)),
        "dst_port": int(rng.choice([80, 443, 53])) if proto != "ICMP" else 0,
        "src_bytes": int(rng.integers(5000, 120000)),
        "dst_bytes": int(rng.integers(0, 7500)),
        "packets": int(rng.integers(180, 1900)),
        "syn_count": int(rng.integers(25, 170)) if proto == "TCP" else 0,
        "failed_logins": 0,
        "connections_per_minute": int(rng.integers(130, 520)),
        "label": 1,
    }


def generate(count_per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [normal_flow(rng) for _ in range(count_per_class)]
    rows.extend(suspicious_flow(rng) for _ in range(count_per_class))
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "synthetic_training.csv"
    sample_path = DATA_DIR / "sample_flows.csv"
    generate(800, seed=42).to_csv(train_path, index=False)
    generate(15, seed=2026).drop(columns=["label"]).to_csv(sample_path, index=False)
    print(f"Created {train_path} (1,600 labeled fictional flow records)")
    print(f"Created {sample_path} (30 unlabeled demo flow records)")
    print("Important: fictional data are for learning, not real-world threat detection.")


if __name__ == "__main__":
    main()
