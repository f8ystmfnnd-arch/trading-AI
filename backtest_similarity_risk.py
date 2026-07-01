"""Backtest Historical Similarity Risk Analyzer.

This script checks whether historical similarity statistics explain future
market outcomes. It is not a direct trading strategy; the risk filter section
is a simple diagnostic for drawdown direction only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SIMILARITY_DIR = BASE_DIR / "data" / "processed" / "similarity"

ALLOWED_PATTERN_LENGTHS = {16, 48, 96, 192}
FEE_RATE = 0.0006

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
    "target_pump_next_4",
    "target_volatility_next_4",
]

RISK_FILTERS = [
    {"name": "hv040_drop007", "high_vol": 0.40, "drop": 0.07},
    {"name": "hv050_drop010", "high_vol": 0.50, "drop": 0.10},
    {"name": "hv060_drop015", "high_vol": 0.60, "drop": 0.15},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest whether similarity-derived risk indicators explain future BTC outcomes."
    )
    parser.add_argument(
        "--pattern-length",
        type=int,
        default=48,
        choices=sorted(ALLOWED_PATTERN_LENGTHS),
        help="Similarity raw dataset length to evaluate.",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Top similar windows per evaluation row.")
    parser.add_argument("--step", type=int, default=16, help="Evaluate every N rows after the initial 30%% warm-up.")
    parser.add_argument(
        "--exclude-recent-days",
        type=float,
        default=7,
        help="Exclude candidates too close to the evaluation timestamp.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="If set, use only this many most recent evaluation points.",
    )
    parser.add_argument("--metric", default="cosine", help="Only cosine is supported in this version.")
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"[error] {message}")
    sys.exit(1)


def input_path_for(pattern_length: int) -> Path:
    return SIMILARITY_DIR / f"BTCUSDT_15m_similarity_raw_L{pattern_length}.csv"


def output_base(pattern_length: int, top_k: int, step: int) -> Path:
    return SIMILARITY_DIR / f"similarity_backtest_L{pattern_length}_top{top_k}_step{step}"


def inspect_columns(path: Path) -> tuple[list[str], list[str]]:
    if not path.exists():
        fail(f"input file not found: {path}")

    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)
    missing_meta = [col for col in METADATA_COLUMNS if col not in columns]
    missing_outcomes = [col for col in REQUIRED_OUTCOME_COLUMNS if col not in columns]
    if missing_meta:
        fail(f"required metadata columns are missing: {missing_meta}")
    if missing_outcomes:
        fail(f"required outcome columns are missing: {missing_outcomes}")

    outcome_columns = REQUIRED_OUTCOME_COLUMNS + [col for col in OPTIONAL_OUTCOME_COLUMNS if col in columns]
    excluded = set(METADATA_COLUMNS + outcome_columns)
    vector_columns = [col for col in columns if col not in excluded]
    if not vector_columns:
        fail("usable vector columns count is 0")
    leaked_targets = [col for col in vector_columns if col.startswith("target_")]
    if leaked_targets:
        fail(f"target columns must not be used as vectors: {leaked_targets[:10]}")
    return outcome_columns, vector_columns


def load_dataset(path: Path, vector_columns: list[str], outcome_columns: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    usecols = METADATA_COLUMNS + outcome_columns + vector_columns
    df = pd.read_csv(path, usecols=usecols)
    if df.empty:
        fail(f"input file has no rows: {path}")

    df["window_start_timestamp"] = pd.to_datetime(df["window_start_timestamp"], utc=True, errors="coerce")
    df["window_end_timestamp"] = pd.to_datetime(df["window_end_timestamp"], utc=True, errors="coerce")
    if df["window_start_timestamp"].isna().any() or df["window_end_timestamp"].isna().any():
        fail("timestamp parsing failed for one or more rows")
    df = df.copy()
    df["_window_end_ns"] = df["window_end_timestamp"].astype("int64")

    matrix = df.loc[:, vector_columns].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(matrix).all():
        fail("vector matrix contains NaN or infinite values")

    norms = np.linalg.norm(matrix, axis=1).astype(np.float32)
    if not np.isfinite(norms).all():
        fail("vector norms contain invalid values")

    for col in REQUIRED_OUTCOME_COLUMNS:
        if df[col].isna().any():
            fail(f"required outcome column contains NaN: {col}")

    return df, matrix, norms


def build_eval_indices(row_count: int, step: int, max_evals: int | None) -> list[int]:
    start_eval_idx = int(row_count * 0.30)
    indices = list(range(start_eval_idx, row_count, step))
    if max_evals is not None:
        if max_evals <= 0:
            fail("--max-evals must be positive when provided")
        indices = indices[-max_evals:]
    if not indices:
        fail("no evaluation indices selected")
    return indices


def top_k_for_eval(
    eval_idx: int,
    df: pd.DataFrame,
    matrix: np.ndarray,
    norms: np.ndarray,
    top_k: int,
    exclude_recent_days: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    eval_time = df.at[eval_idx, "window_end_timestamp"]
    cutoff_time = eval_time - pd.Timedelta(days=exclude_recent_days)
    cutoff_idx = int(np.searchsorted(df["_window_end_ns"].to_numpy(), cutoff_time.value, side="left"))

    if cutoff_idx <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0

    candidate_indices = np.arange(cutoff_idx, dtype=np.int64)
    candidate_norms = norms[:cutoff_idx]
    valid_mask = candidate_norms > 0
    if not valid_mask.all():
        candidate_indices = candidate_indices[valid_mask]
        candidate_norms = candidate_norms[valid_mask]

    candidate_count = len(candidate_indices)
    if candidate_count < top_k:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32), candidate_count

    current = matrix[eval_idx]
    current_norm = norms[eval_idx]
    if current_norm <= 0 or not math.isfinite(float(current_norm)):
        fail(f"eval_idx {eval_idx} has zero or invalid vector norm")

    sims = (matrix[candidate_indices] @ current) / (candidate_norms * current_norm)
    if not np.isfinite(sims).all():
        fail(f"similarity contains NaN or infinite values at eval_idx {eval_idx}")

    pick = np.argpartition(sims, -top_k)[-top_k:]
    order = np.argsort(sims[pick])[::-1]
    selected_positions = pick[order]
    return candidate_indices[selected_positions], sims[selected_positions], candidate_count


def eval_one(
    eval_idx: int,
    df: pd.DataFrame,
    matrix: np.ndarray,
    norms: np.ndarray,
    top_k: int,
    exclude_recent_days: float,
) -> dict[str, Any] | None:
    top_indices, top_sims, candidate_count = top_k_for_eval(
        eval_idx=eval_idx,
        df=df,
        matrix=matrix,
        norms=norms,
        top_k=top_k,
        exclude_recent_days=exclude_recent_days,
    )
    if candidate_count < top_k or len(top_indices) < top_k:
        return None

    top = df.iloc[top_indices]
    row: dict[str, Any] = {
        "eval_window_start_timestamp": str(df.at[eval_idx, "window_start_timestamp"]),
        "eval_window_end_timestamp": str(df.at[eval_idx, "window_end_timestamp"]),
        "pattern_length": int(df.at[eval_idx, "pattern_length"]),
        "top_k": top_k,
        "candidate_count": candidate_count,
        "mean_similarity": float(np.mean(top_sims)),
        "min_similarity": float(np.min(top_sims)),
        "max_similarity": float(np.max(top_sims)),
        "similar_up_ratio": float(top["target_up_next"].mean()),
        "similar_avg_return_next": float(top["target_return_next"].mean()),
        "similar_avg_return_next_4": float(top["target_return_next_4"].mean()),
        "similar_high_vol_ratio": float(top["target_volatility_high_next_4"].mean()),
        "similar_drop_ratio": float(top["target_drop_next_4"].mean()),
        "similar_big_move_ratio": float(top["target_big_move_next_4"].mean()),
        "actual_target_up_next": int(df.at[eval_idx, "target_up_next"]),
        "actual_return_next": float(df.at[eval_idx, "target_return_next"]),
        "actual_return_next_4": float(df.at[eval_idx, "target_return_next_4"]),
        "actual_high_vol": int(df.at[eval_idx, "target_volatility_high_next_4"]),
        "actual_drop": int(df.at[eval_idx, "target_drop_next_4"]),
        "actual_big_move": int(df.at[eval_idx, "target_big_move_next_4"]),
    }

    if "target_pump_next_4" in df.columns:
        row["similar_pump_ratio"] = float(top["target_pump_next_4"].mean())
        row["actual_pump"] = int(df.at[eval_idx, "target_pump_next_4"])
    if "target_volatility_next_4" in df.columns:
        row["similar_volatility_next_4_mean"] = float(top["target_volatility_next_4"].mean())

    return row


def pearson_corr(left: pd.Series, right: pd.Series) -> float | None:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return None
    value = left.corr(right, method="pearson")
    if pd.isna(value):
        return None
    return float(value)


def spearman_corr(left: pd.Series, right: pd.Series) -> float | None:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return None
    value = left.rank().corr(right.rank(), method="pearson")
    if pd.isna(value):
        return None
    return float(value)


def correlation_summary(results: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    pairs = {
        "avg_return_next_4_vs_actual_return_next_4": ("similar_avg_return_next_4", "actual_return_next_4"),
        "up_ratio_vs_actual_up": ("similar_up_ratio", "actual_target_up_next"),
        "high_vol_ratio_vs_actual_high_vol": ("similar_high_vol_ratio", "actual_high_vol"),
        "drop_ratio_vs_actual_drop": ("similar_drop_ratio", "actual_drop"),
        "big_move_ratio_vs_actual_big_move": ("similar_big_move_ratio", "actual_big_move"),
    }
    output: dict[str, dict[str, float | None]] = {}
    for name, (left, right) in pairs.items():
        output[name] = {
            "pearson": pearson_corr(results[left], results[right]),
            "spearman": spearman_corr(results[left], results[right]),
        }
    return output


def bucket_table(
    results: pd.DataFrame,
    source_col: str,
    bins: list[float],
    labels: list[str],
    aggregations: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    bucket = pd.cut(results[source_col], bins=bins, labels=labels, include_lowest=True, right=False)
    temp = results.copy()
    temp["bucket"] = bucket.astype(str)
    rows = []
    for label in labels:
        part = temp.loc[temp["bucket"] == label]
        row: dict[str, Any] = {"bucket": label, "count": int(len(part))}
        for output_name, (column, func) in aggregations.items():
            row[output_name] = None if part.empty else float(getattr(part[column], func)())
        rows.append(row)
    return rows


def bucket_summary(results: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    high_vol = bucket_table(
        results,
        "similar_high_vol_ratio",
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.000001],
        labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        aggregations={
            "actual_high_vol_rate": ("actual_high_vol", "mean"),
            "avg_actual_return_next_4": ("actual_return_next_4", "mean"),
            "avg_actual_drop_rate": ("actual_drop", "mean"),
            "avg_actual_big_move_rate": ("actual_big_move", "mean"),
        },
    )
    drop = bucket_table(
        results,
        "similar_drop_ratio",
        bins=[0.0, 0.03, 0.07, 0.15, 1.000001],
        labels=["0.00-0.03", "0.03-0.07", "0.07-0.15", "0.15-1.00"],
        aggregations={
            "actual_drop_rate": ("actual_drop", "mean"),
            "actual_high_vol_rate": ("actual_high_vol", "mean"),
            "avg_actual_return_next_4": ("actual_return_next_4", "mean"),
            "actual_big_move_rate": ("actual_big_move", "mean"),
        },
    )
    up = bucket_table(
        results,
        "similar_up_ratio",
        bins=[0.0, 0.4, 0.5, 0.6, 1.000001],
        labels=["0.0-0.4", "0.4-0.5", "0.5-0.6", "0.6-1.0"],
        aggregations={
            "actual_up_rate": ("actual_target_up_next", "mean"),
            "avg_actual_return_next_4": ("actual_return_next_4", "mean"),
            "actual_high_vol_rate": ("actual_high_vol", "mean"),
            "actual_drop_rate": ("actual_drop", "mean"),
        },
    )
    return {"high_vol": high_vol, "drop": drop, "up": up}


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def risk_filter_summary(results: pd.DataFrame) -> list[dict[str, Any]]:
    returns = results["actual_return_next_4"].astype(float)
    buy_and_hold_equity = (1.0 + returns).cumprod()
    buy_and_hold_return = float(buy_and_hold_equity.iloc[-1] - 1.0)
    buy_and_hold_mdd = max_drawdown(buy_and_hold_equity)
    output = []

    for spec in RISK_FILTERS:
        raw_position = ~(
            (results["similar_high_vol_ratio"] >= spec["high_vol"])
            | (results["similar_drop_ratio"] >= spec["drop"])
        )
        position = raw_position.astype(float).shift(1).fillna(1.0)
        position_change = position.diff().abs().fillna(0.0)
        fees = position_change * FEE_RATE
        strategy_return = position * returns - fees
        equity = (1.0 + strategy_return).cumprod()
        risk_off = position < 1.0

        risk_off_part = results.loc[risk_off]
        output.append(
            {
                "filter_name": spec["name"],
                "threshold_high_vol": spec["high_vol"],
                "threshold_drop": spec["drop"],
                "total_return": float(equity.iloc[-1] - 1.0),
                "buy_and_hold_return": buy_and_hold_return,
                "max_drawdown": max_drawdown(equity),
                "buy_and_hold_max_drawdown": buy_and_hold_mdd,
                "exposure_ratio": float(position.mean()),
                "risk_off_ratio": float(risk_off.mean()),
                "number_of_position_changes": int((position_change > 0).sum()),
                "total_fee_paid": float(fees.sum()),
                "avg_return_when_risk_on": float(results.loc[~risk_off, "actual_return_next_4"].mean()),
                "avg_return_when_risk_off": None
                if risk_off_part.empty
                else float(risk_off_part["actual_return_next_4"].mean()),
                "actual_high_vol_rate_when_risk_off": None
                if risk_off_part.empty
                else float(risk_off_part["actual_high_vol"].mean()),
                "actual_drop_rate_when_risk_off": None
                if risk_off_part.empty
                else float(risk_off_part["actual_drop"].mean()),
            }
        )
    return output


def monotonic_direction(rows: list[dict[str, Any]], value_key: str) -> str:
    values = [row[value_key] for row in rows if row["count"] > 0 and row[value_key] is not None]
    if len(values) < 2:
        return "insufficient"
    increases = sum(values[i] <= values[i + 1] for i in range(len(values) - 1))
    decreases = sum(values[i] >= values[i + 1] for i in range(len(values) - 1))
    if increases == len(values) - 1:
        return "increasing"
    if decreases == len(values) - 1:
        return "decreasing"
    return "mixed"


def interpretation(bucket: dict[str, list[dict[str, Any]]], risk_filters: list[dict[str, Any]]) -> list[str]:
    messages = []
    high_vol_trend = monotonic_direction(bucket["high_vol"], "actual_high_vol_rate")
    drop_trend = monotonic_direction(bucket["drop"], "actual_drop_rate")
    mdd_improved = any(row["max_drawdown"] > row["buy_and_hold_max_drawdown"] for row in risk_filters)

    if high_vol_trend == "increasing":
        messages.append("similar_high_vol_ratio shows useful monotonic risk signal.")
    else:
        messages.append("similar_high_vol_ratio does not show clean monotonic separation.")

    if drop_trend == "increasing":
        messages.append("similar_drop_ratio shows useful drop-risk separation.")
    else:
        messages.append("similar_drop_ratio does not show stable drop-risk separation.")

    if mdd_improved:
        messages.append("simple similarity risk filter improved drawdown in at least one threshold set.")
    else:
        messages.append("simple similarity risk filter did not improve drawdown.")

    messages.append("Risk filter results are row-sampled diagnostics, not a production trading backtest.")
    return messages


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n[{title}]")
    if not rows:
        print("(empty)")
        return
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.metric != "cosine":
        fail("only --metric cosine is supported")
    if args.top_k <= 0:
        fail("--top-k must be positive")
    if args.step <= 0:
        fail("--step must be positive")
    if args.exclude_recent_days < 0:
        fail("--exclude-recent-days must be non-negative")

    path = input_path_for(args.pattern_length)
    print("[purpose] Backtesting Historical Similarity Risk Analyzer")
    print("[purpose] This is not a direct trading strategy.")
    print(f"[input] {path}")
    print(
        f"[config] pattern_length={args.pattern_length} top_k={args.top_k} "
        f"step={args.step} exclude_recent_days={args.exclude_recent_days} max_evals={args.max_evals}"
    )

    outcome_columns, vector_columns = inspect_columns(path)
    df, matrix, norms = load_dataset(path, vector_columns, outcome_columns)
    eval_indices = build_eval_indices(len(df), args.step, args.max_evals)

    print(f"[input] rows={len(df):,}")
    print(f"[vectors] vector_column_count={len(vector_columns):,}")
    print(f"[eval] selected_eval_count={len(eval_indices):,}")
    print(f"[eval] period_start={df.at[eval_indices[0], 'window_end_timestamp']}")
    print(f"[eval] period_end={df.at[eval_indices[-1], 'window_end_timestamp']}")

    rows = []
    skipped = 0
    for count, eval_idx in enumerate(eval_indices, start=1):
        row = eval_one(eval_idx, df, matrix, norms, args.top_k, args.exclude_recent_days)
        if row is None:
            skipped += 1
        else:
            rows.append(row)
        if count % 100 == 0 or count == len(eval_indices):
            print(f"[progress] evals={count:,}/{len(eval_indices):,} saved={len(rows):,} skipped={skipped:,}")

    if not rows:
        fail("no evaluation rows were produced")

    results = pd.DataFrame(rows)
    detail_path = output_base(args.pattern_length, args.top_k, args.step).with_suffix(".csv")
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(detail_path, index=False)

    corr = correlation_summary(results)
    buckets = bucket_summary(results)
    risk_filters = risk_filter_summary(results)
    interp = interpretation(buckets, risk_filters)
    summary = {
        "pattern_length": args.pattern_length,
        "top_k": args.top_k,
        "step": args.step,
        "exclude_recent_days": args.exclude_recent_days,
        "max_evals": args.max_evals,
        "eval_count": int(len(results)),
        "skipped_eval_count": int(skipped),
        "period_start": str(results["eval_window_end_timestamp"].iloc[0]),
        "period_end": str(results["eval_window_end_timestamp"].iloc[-1]),
        "vector_column_count": int(len(vector_columns)),
        "correlation_summary": corr,
        "bucket_summary": buckets,
        "risk_filter_summary": risk_filters,
        "interpretation": interp,
    }

    summary_path = output_base(args.pattern_length, args.top_k, args.step)
    if args.max_evals is not None:
        summary_path = summary_path.with_name(summary_path.name + f"_max{args.max_evals}")
    summary_json_path = summary_path.with_name(summary_path.name + "_summary.json")
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[saved] detailed CSV: {detail_path}")
    print(f"[saved] summary JSON: {summary_json_path}")
    print("\n[correlation summary]")
    print(json.dumps(corr, indent=2, ensure_ascii=False))
    print_table("high_vol buckets", buckets["high_vol"])
    print_table("drop buckets", buckets["drop"])
    print_table("up buckets", buckets["up"])
    print_table("simple risk filter summary", risk_filters)
    print("\n[interpretation]")
    for message in interp:
        print(f"- {message}")
    print("\n[note] Fees are included, but this is a row-sampled diagnostic backtest, not an execution-grade strategy simulation.")


if __name__ == "__main__":
    main()
