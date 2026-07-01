"""Historical Similarity Risk Analyzer.

This script compares the latest BTC market pattern with historical pattern
vectors and summarizes what happened after the most similar past windows.

Important: this is not a direct buy/sell signal. It is an explainable risk
reference for BTC Market Regime & Risk Guard AI.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "processed" / "similarity"
MODEL_DIR = BASE_DIR / "model"

ALLOWED_PATTERN_LENGTHS = {16, 48, 96, 192}
CHUNK_SIZE = 5_000

METADATA_COLUMNS = [
    "window_start_timestamp",
    "window_end_timestamp",
    "pattern_length",
]

REQUIRED_OUTCOME_COLUMNS = [
    "target_return_next",
    "target_up_next",
    "target_return_next_4",
    "target_volatility_high_next_4",
    "target_drop_next_4",
    "target_big_move_next_4",
]

OPTIONAL_OUTCOME_COLUMNS = [
    "target_abs_return_next_4",
    "target_volatility_next_4",
    "target_pump_next_4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find historical BTC patterns similar to the latest pattern and summarize future outcomes."
    )
    parser.add_argument(
        "--pattern-length",
        type=int,
        default=48,
        choices=sorted(ALLOWED_PATTERN_LENGTHS),
        help="Pattern length to analyze. L48 means recent 12 hours on 15m bars.",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Number of similar historical windows to keep.")
    parser.add_argument(
        "--exclude-recent-days",
        type=float,
        default=7,
        help="Exclude candidates whose window end is within this many days of the latest window.",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        help="Similarity metric. First version supports only cosine.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV and JSON outputs.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"[error] {message}")
    sys.exit(1)


def input_path_for(pattern_length: int) -> Path:
    return DEFAULT_OUTPUT_DIR / f"BTCUSDT_15m_similarity_raw_L{pattern_length}.csv"


def inspect_columns(path: Path) -> tuple[list[str], list[str]]:
    try:
        header = pd.read_csv(path, nrows=0)
    except FileNotFoundError:
        fail(f"input file not found: {path}")
    except Exception as exc:
        fail(f"failed to read input header: {exc}")

    columns = list(header.columns)
    missing_metadata = [col for col in METADATA_COLUMNS if col not in columns]
    if missing_metadata:
        fail(f"required metadata columns are missing: {missing_metadata}")

    missing_outcomes = [col for col in REQUIRED_OUTCOME_COLUMNS if col not in columns]
    if missing_outcomes:
        fail(f"required outcome columns are missing: {missing_outcomes}")

    outcome_columns = REQUIRED_OUTCOME_COLUMNS + [col for col in OPTIONAL_OUTCOME_COLUMNS if col in columns]
    excluded = set(METADATA_COLUMNS + outcome_columns)
    vector_columns = [col for col in columns if col not in excluded]
    if not vector_columns:
        fail("usable vector columns count is 0")

    leaked_targets = [col for col in vector_columns if col.startswith("target_")]
    if leaked_targets:
        fail(f"target columns must not be used as pattern vectors: {leaked_targets[:10]}")

    return outcome_columns, vector_columns


def read_latest_row(path: Path) -> tuple[pd.Series, int]:
    latest_row: pd.Series | None = None
    row_count = 0
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        if chunk.empty:
            continue
        row_count += len(chunk)
        latest_row = chunk.iloc[-1].copy()

    if latest_row is None:
        fail(f"input file has no rows: {path}")

    return latest_row, row_count


def row_vector(row: pd.Series, vector_columns: list[str]) -> np.ndarray:
    vector = row.loc[vector_columns].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(vector).all():
        fail("latest pattern vector contains NaN or infinite values")
    norm = np.linalg.norm(vector)
    if norm == 0 or not math.isfinite(norm):
        fail("latest pattern vector has zero or invalid norm")
    return vector


def top_k_similar(
    path: Path,
    vector_columns: list[str],
    outcome_columns: list[str],
    current_vector: np.ndarray,
    cutoff_time: pd.Timestamp,
    top_k: int,
) -> tuple[pd.DataFrame, int]:
    heap: list[tuple[float, int, dict[str, Any]]] = []
    candidate_count = 0
    current_norm = np.linalg.norm(current_vector)
    output_columns = METADATA_COLUMNS + outcome_columns

    for chunk_index, chunk in enumerate(pd.read_csv(path, chunksize=CHUNK_SIZE), start=1):
        chunk["window_end_timestamp"] = pd.to_datetime(chunk["window_end_timestamp"], utc=True, errors="coerce")
        if chunk["window_end_timestamp"].isna().any():
            fail(f"invalid window_end_timestamp found in chunk {chunk_index}")

        candidates = chunk.loc[chunk["window_end_timestamp"] < cutoff_time].copy()
        if candidates.empty:
            continue

        matrix = candidates.loc[:, vector_columns].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(matrix).all():
            fail(f"candidate vector contains NaN or infinite values in chunk {chunk_index}")

        norms = np.linalg.norm(matrix, axis=1)
        valid_mask = norms > 0
        if not valid_mask.all():
            candidates = candidates.loc[valid_mask].copy()
            matrix = matrix[valid_mask]
            norms = norms[valid_mask]

        if candidates.empty:
            continue

        similarities = matrix @ current_vector / (norms * current_norm)
        if not np.isfinite(similarities).all():
            fail(f"similarity calculation produced NaN or infinite values in chunk {chunk_index}")

        candidate_count += len(candidates)
        metadata = candidates.loc[:, output_columns].copy()
        metadata["window_end_timestamp"] = metadata["window_end_timestamp"].astype(str)

        for local_index, (similarity, row_dict) in enumerate(
            zip(similarities, metadata.to_dict(orient="records"))
        ):
            item = (float(similarity), chunk_index * CHUNK_SIZE + local_index, row_dict)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif similarity > heap[0][0]:
                heapq.heapreplace(heap, item)

    if candidate_count < top_k:
        fail(f"candidate_count ({candidate_count}) is smaller than top_k ({top_k})")

    if len(heap) < top_k:
        fail(f"only {len(heap)} valid similar rows found, expected {top_k}")

    rows = []
    for rank, (similarity, _, row_dict) in enumerate(sorted(heap, key=lambda x: x[0], reverse=True), start=1):
        rows.append({"rank": rank, "similarity": similarity, **row_dict})

    result = pd.DataFrame(rows)
    if result["similarity"].isna().any():
        fail("output similarity contains NaN")
    if result["rank"].tolist() != list(range(1, top_k + 1)):
        fail("output ranks are not 1..top_k")

    return result, candidate_count


def risk_level(value: float, high_cutoff: float, medium_cutoff: float) -> str:
    if value >= high_cutoff:
        return "HIGH"
    if value >= medium_cutoff:
        return "MEDIUM"
    return "LOW"


def direction_hint(up_ratio: float) -> str:
    if up_ratio >= 0.60:
        return "BULLISH_HINT"
    if up_ratio <= 0.40:
        return "BEARISH_HINT"
    return "NO_CLEAR_EDGE"


def action_hint(high_vol_level: str, drop_risk_level: str) -> str:
    if high_vol_level == "HIGH" and drop_risk_level in {"MEDIUM", "HIGH"}:
        return "NO_TRADE"
    if high_vol_level == "HIGH":
        return "CAUTION"
    if drop_risk_level == "MEDIUM":
        return "CAUTION"
    return "NORMAL"


def summarize(
    top_df: pd.DataFrame,
    pattern_length: int,
    top_k: int,
    metric: str,
    exclude_recent_days: float,
    current_row: pd.Series,
    candidate_count: int,
    vector_column_count: int,
) -> dict[str, Any]:
    similar_up_ratio = float(top_df["target_up_next"].mean())
    similar_high_vol_ratio = float(top_df["target_volatility_high_next_4"].mean())
    similar_drop_ratio = float(top_df["target_drop_next_4"].mean())
    similar_big_move_ratio = float(top_df["target_big_move_next_4"].mean())
    high_vol = risk_level(similar_high_vol_ratio, high_cutoff=0.60, medium_cutoff=0.40)
    drop_risk = risk_level(similar_drop_ratio, high_cutoff=0.15, medium_cutoff=0.07)
    direction = direction_hint(similar_up_ratio)
    action = action_hint(high_vol, drop_risk)

    summary: dict[str, Any] = {
        "purpose": "Historical Similarity Risk Analyzer",
        "note": "This is not a direct buy/sell signal. It is an explainable risk reference.",
        "pattern_length": pattern_length,
        "top_k": top_k,
        "metric": metric,
        "exclude_recent_days": exclude_recent_days,
        "current_window_start_timestamp": str(current_row["window_start_timestamp"]),
        "current_window_end_timestamp": str(current_row["window_end_timestamp"]),
        "candidate_count": int(candidate_count),
        "vector_column_count": int(vector_column_count),
        "mean_similarity": float(top_df["similarity"].mean()),
        "min_similarity": float(top_df["similarity"].min()),
        "max_similarity": float(top_df["similarity"].max()),
        "similar_up_ratio": similar_up_ratio,
        "similar_avg_return_next": float(top_df["target_return_next"].mean()),
        "similar_median_return_next": float(top_df["target_return_next"].median()),
        "similar_avg_return_next_4": float(top_df["target_return_next_4"].mean()),
        "similar_median_return_next_4": float(top_df["target_return_next_4"].median()),
        "similar_high_vol_ratio": similar_high_vol_ratio,
        "similar_drop_ratio": similar_drop_ratio,
        "similar_big_move_ratio": similar_big_move_ratio,
        "high_vol_level": high_vol,
        "drop_risk_level": drop_risk,
        "direction_hint": direction,
        "action_hint": action,
    }

    if "target_pump_next_4" in top_df.columns:
        summary["similar_pump_ratio"] = float(top_df["target_pump_next_4"].mean())
    if "target_volatility_next_4" in top_df.columns:
        summary["similar_volatility_next_4_mean"] = float(top_df["target_volatility_next_4"].mean())

    return summary


def plot_top5_return_pattern(
    top_df: pd.DataFrame,
    path: Path,
    current_row: pd.Series,
    vector_columns: list[str],
    pattern_length: int,
    top_k: int,
    output_path: Path,
) -> bool:
    return_cols = sorted(
        [col for col in vector_columns if col.startswith("return_1_lag_")],
        key=lambda col: int(col.rsplit("_", 1)[-1]),
        reverse=True,
    )
    if not return_cols:
        print("[warning] return_1 lag columns not found. Skipping plot.")
        return False

    top_end_times = set(top_df.head(5)["window_end_timestamp"].astype(str))
    rows = []
    usecols = ["window_end_timestamp"] + return_cols
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE):
        matched = chunk.loc[chunk["window_end_timestamp"].astype(str).isin(top_end_times)]
        if not matched.empty:
            rows.append(matched)
        if sum(len(frame) for frame in rows) >= min(5, len(top_end_times)):
            break

    if not rows:
        print("[warning] top 5 rows for plot were not found. Skipping plot.")
        return False

    top_patterns = pd.concat(rows, ignore_index=True)
    current_sequence = current_row.loc[return_cols].to_numpy(dtype=np.float64)
    x = np.arange(len(return_cols))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    plt.plot(x, current_sequence, label="current", linewidth=2.8, color="black")

    for rank, row in top_df.head(5).iterrows():
        matched = top_patterns.loc[top_patterns["window_end_timestamp"].astype(str) == str(row["window_end_timestamp"])]
        if matched.empty:
            continue
        sequence = matched.iloc[0].loc[return_cols].to_numpy(dtype=np.float64)
        plt.plot(x, sequence, alpha=0.7, linewidth=1.4, label=f"rank {int(row['rank'])}: {row['window_end_timestamp']}")

    plt.title(f"Top 5 Similar Return Patterns | L{pattern_length} top{top_k} | {current_row['window_end_timestamp']}")
    plt.xlabel("Lag index, oldest to current")
    plt.ylabel("Normalized return_1")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def save_outputs(
    top_df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    pattern_length: int,
    top_k: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"similar_patterns_latest_L{pattern_length}_top{top_k}.csv"
    json_path = output_dir / f"similar_patterns_latest_L{pattern_length}_top{top_k}_summary.json"
    top_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def print_summary(summary: dict[str, Any], top_df: pd.DataFrame, paths: dict[str, Path]) -> None:
    print("\n[purpose] Historical Similarity Risk Analyzer")
    print("[purpose] This is not a direct buy/sell signal.")
    print("\n[summary]")
    for key in [
        "mean_similarity",
        "min_similarity",
        "max_similarity",
        "similar_up_ratio",
        "similar_avg_return_next_4",
        "similar_high_vol_ratio",
        "similar_drop_ratio",
        "similar_big_move_ratio",
        "high_vol_level",
        "drop_risk_level",
        "direction_hint",
        "action_hint",
    ]:
        print(f"{key}: {summary[key]}")

    print("\n[top 10 similar windows]")
    table_columns = [
        "rank",
        "similarity",
        "window_end_timestamp",
        "target_return_next_4",
        "target_volatility_high_next_4",
        "target_drop_next_4",
        "target_big_move_next_4",
    ]
    print(top_df.loc[:, table_columns].head(10).to_string(index=False))

    print("\n[saved paths]")
    for label, path in paths.items():
        print(f"{label}: {path}")


def main() -> None:
    args = parse_args()
    if args.metric != "cosine":
        fail("only --metric cosine is supported in the first version")
    if args.top_k <= 0:
        fail("--top-k must be positive")
    if args.exclude_recent_days < 0:
        fail("--exclude-recent-days must be non-negative")

    path = input_path_for(args.pattern_length)
    output_dir = Path(args.output_dir)

    print("[purpose] Historical Similarity Risk Analyzer")
    print("[purpose] This is not a direct buy/sell signal.")
    print(f"[input] {path}")
    print(f"[config] pattern_length={args.pattern_length}")
    print(f"[config] top_k={args.top_k}")
    print(f"[config] exclude_recent_days={args.exclude_recent_days}")
    print(f"[config] metric={args.metric}")

    outcome_columns, vector_columns = inspect_columns(path)
    current_row, row_count = read_latest_row(path)
    if row_count <= args.top_k:
        fail(f"rows ({row_count}) must be greater than top_k ({args.top_k})")

    current_end_time = pd.to_datetime(current_row["window_end_timestamp"], utc=True, errors="coerce")
    if pd.isna(current_end_time):
        fail("latest row has invalid window_end_timestamp")
    cutoff_time = current_end_time - pd.Timedelta(days=args.exclude_recent_days)
    current_vector = row_vector(current_row, vector_columns)

    print(f"[input] rows={row_count:,}")
    print(f"[current] window_start={current_row['window_start_timestamp']}")
    print(f"[current] window_end={current_row['window_end_timestamp']}")
    print(f"[filter] cutoff_time={cutoff_time}")
    print(f"[vectors] vector_column_count={len(vector_columns):,}")
    print("[similarity] calculating cosine similarities by chunk...")

    top_df, candidate_count = top_k_similar(
        path=path,
        vector_columns=vector_columns,
        outcome_columns=outcome_columns,
        current_vector=current_vector,
        cutoff_time=cutoff_time,
        top_k=args.top_k,
    )

    print(f"[similarity] complete. candidate rows after recent exclusion={candidate_count:,}")
    if len(top_df) != args.top_k:
        fail(f"output Top K rows ({len(top_df)}) does not equal top_k ({args.top_k})")

    summary = summarize(
        top_df=top_df,
        pattern_length=args.pattern_length,
        top_k=args.top_k,
        metric=args.metric,
        exclude_recent_days=args.exclude_recent_days,
        current_row=current_row,
        candidate_count=candidate_count,
        vector_column_count=len(vector_columns),
    )

    csv_path, json_path = save_outputs(top_df, summary, output_dir, args.pattern_length, args.top_k)
    png_path = MODEL_DIR / f"similar_patterns_latest_L{args.pattern_length}_top5.png"
    plot_created = plot_top5_return_pattern(
        top_df=top_df,
        path=path,
        current_row=current_row,
        vector_columns=vector_columns,
        pattern_length=args.pattern_length,
        top_k=args.top_k,
        output_path=png_path,
    )

    paths = {"csv": csv_path, "json": json_path}
    if plot_created:
        paths["png"] = png_path
    print_summary(summary, top_df, paths)


if __name__ == "__main__":
    main()
