"""
Create historical similarity pattern datasets for BTC Market Regime & Risk Guard AI.

This script does not train a model. It creates window-based datasets that can be
used later to find historical chart patterns similar to the current market.

Input:
    data/processed/BTCUSDT_15m_risk_targets.csv

Outputs:
    data/processed/similarity/BTCUSDT_15m_similarity_summary_L{length}.csv
    data/processed/similarity/BTCUSDT_15m_similarity_raw_L{length}.csv
    data/processed/similarity_feature_scaler.csv

Run:
    py -3 create_similarity_dataset.py
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.scaling import fit_training_standardizer


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_risk_targets.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "similarity"
SCALER_PATH = BASE_DIR / "data" / "processed" / "similarity_feature_scaler.csv"

PATTERN_LENGTHS = [16, 48, 96, 192]
RAW_CHUNK_ROWS = 2_000

CANDIDATE_FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "volatility_20",
    "volatility_60",
    "high_low_range",
    "volume_change_1",
    "close_open_return",
    "ma_distance_20",
    "ma_distance_60",
    "rsi_14",
    "h1_return_1",
    "h1_ma_distance_20",
    "h4_return_1",
    "h4_ma_distance_20",
]

REQUIRED_COLUMNS = [
    "timestamp",
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

METADATA_COLUMNS = ["window_start_timestamp", "window_end_timestamp", "pattern_length"]
SUMMARY_STATS = ["mean", "std", "min", "max", "last", "first", "change"]


def load_input() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file does not exist: {INPUT_PATH}")

    print(f"[input] {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"[input] rows={len(df):,}")

    missing_required = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required timestamp/outcome columns: {missing_required}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("timestamp column contains invalid values")

    before_rows = len(df)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    removed_duplicates = before_rows - len(df)
    print(f"[input] duplicate timestamps removed before dataset creation: {removed_duplicates:,}")
    print(f"[input] period start={df['timestamp'].iloc[0]} end={df['timestamp'].iloc[-1]}")
    return df


def default_training_fit_end(df: pd.DataFrame) -> pd.Timestamp:
    train_end = int(len(df) * 0.70)
    if train_end <= 0:
        raise ValueError("Not enough rows to define the training scaler interval")
    return pd.to_datetime(df["timestamp"].iloc[train_end - 1], utc=True)


def saved_scaler_fit_end(scaler_path: Path) -> pd.Timestamp | None:
    if not scaler_path.exists():
        return None
    try:
        metadata = pd.read_csv(scaler_path, nrows=1)
    except (OSError, pd.errors.ParserError):
        return None
    if "scaler_version" not in metadata.columns or "fit_end" not in metadata.columns or metadata.empty:
        return None
    value = pd.to_datetime(metadata.loc[0, "fit_end"], utc=True, errors="coerce")
    return None if pd.isna(value) else value


def select_and_scale_features(
    df: pd.DataFrame,
    fit_end: object | None = None,
    *,
    run_id: str = "similarity",
    scaler_path: Path = SCALER_PATH,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    existing_features = [column for column in CANDIDATE_FEATURE_COLUMNS if column in df.columns]
    missing_features = [column for column in CANDIDATE_FEATURE_COLUMNS if column not in df.columns]
    if not existing_features:
        raise ValueError("No usable pattern feature columns were found")

    cutoff = fit_end
    if cutoff is None:
        cutoff = saved_scaler_fit_end(scaler_path)
    if cutoff is None:
        cutoff = default_training_fit_end(df)
    scaler, scaler_excluded = fit_training_standardizer(
        df,
        existing_features,
        cutoff,
        run_id=run_id,
    )
    valid_features = list(scaler.feature_names)
    excluded_features = [*missing_features, *scaler_excluded]
    scaled = scaler.transform(df)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    scaler.metadata_frame().to_csv(scaler_path, index=False)

    if scaled.isna().any().any():
        nan_counts = scaled.isna().sum()
        nan_counts = nan_counts[nan_counts > 0]
        raise ValueError(f"Scaled feature data contains NaN values: {nan_counts.to_dict()}")

    print("[features] used feature columns:")
    for feature in valid_features:
        print(f"  - {feature}")
    print("[features] excluded feature columns:")
    for feature in excluded_features:
        print(f"  - {feature}")
    print(f"[features] total used={len(valid_features):,}")
    print(f"[scaler] fit period: {scaler.fit_start} ~ {scaler.fit_end}")
    print(f"[scaler] saved: {scaler_path}")
    return scaled, valid_features, excluded_features


def outcome_columns(df: pd.DataFrame) -> list[str]:
    return [*REQUIRED_COLUMNS[1:], *[column for column in OPTIONAL_OUTCOME_COLUMNS if column in df.columns]]


def build_base_metadata(df: pd.DataFrame, length: int, outcomes: list[str]) -> pd.DataFrame:
    start_index = length - 1
    metadata = pd.DataFrame(
        {
            "window_start_timestamp": df["timestamp"].iloc[: len(df) - length + 1].to_numpy(),
            "window_end_timestamp": df["timestamp"].iloc[start_index:].to_numpy(),
            "pattern_length": length,
        }
    )
    for column in outcomes:
        metadata[column] = df[column].iloc[start_index:].to_numpy()
    return metadata


def create_summary_dataset(
    scaled: pd.DataFrame,
    source_df: pd.DataFrame,
    feature_columns: list[str],
    outcomes: list[str],
    length: int,
) -> pd.DataFrame:
    metadata = build_base_metadata(source_df, length, outcomes)
    summary_parts = [metadata]

    for feature in feature_columns:
        series = scaled[feature]
        rolling = series.rolling(window=length)
        stats = pd.DataFrame(
            {
                f"{feature}_mean": rolling.mean().iloc[length - 1 :].to_numpy(),
                f"{feature}_std": rolling.std(ddof=0).iloc[length - 1 :].to_numpy(),
                f"{feature}_min": rolling.min().iloc[length - 1 :].to_numpy(),
                f"{feature}_max": rolling.max().iloc[length - 1 :].to_numpy(),
                f"{feature}_last": series.iloc[length - 1 :].to_numpy(),
                f"{feature}_first": series.iloc[: len(series) - length + 1].to_numpy(),
            }
        )
        stats[f"{feature}_change"] = stats[f"{feature}_last"] - stats[f"{feature}_first"]
        summary_parts.append(stats)

    summary_df = pd.concat(summary_parts, axis=1)
    return summary_df


def raw_columns(feature_columns: list[str], length: int) -> list[str]:
    columns: list[str] = []
    for feature in feature_columns:
        for lag in range(length - 1, -1, -1):
            columns.append(f"{feature}_lag_{lag}")
    return columns


def create_raw_dataset(
    scaled: pd.DataFrame,
    source_df: pd.DataFrame,
    feature_columns: list[str],
    outcomes: list[str],
    length: int,
    output_path: Path,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        prefix=f".{output_path.name}.",
        dir=output_path.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        shape = _write_raw_dataset(
            scaled,
            source_df,
            feature_columns,
            outcomes,
            length,
            temp_path,
        )
        validate_saved_csv_streaming(
            temp_path,
            outcomes,
            length,
            "temporary raw",
            expected_shape=shape,
        )
        temp_path.replace(output_path)
        return shape
    finally:
        temp_path.unlink(missing_ok=True)


def _write_raw_dataset(
    scaled: pd.DataFrame,
    source_df: pd.DataFrame,
    feature_columns: list[str],
    outcomes: list[str],
    length: int,
    output_path: Path,
) -> tuple[int, int]:
    values = scaled[feature_columns].to_numpy(dtype=np.float32, copy=True)
    metadata = build_base_metadata(source_df, length, outcomes)
    raw_feature_columns = raw_columns(feature_columns, length)
    expected_rows = len(metadata)

    wrote_header = False
    for start in range(0, expected_rows, RAW_CHUNK_ROWS):
        stop = min(start + RAW_CHUNK_ROWS, expected_rows)
        chunk_rows = stop - start
        chunk_arrays = []
        for feature_index in range(len(feature_columns)):
            feature_windows = np.lib.stride_tricks.sliding_window_view(
                values[:, feature_index], length
            )[start:stop]
            chunk_arrays.append(np.asarray(feature_windows, dtype=np.float32))
        raw_matrix = np.concatenate(chunk_arrays, axis=1)

        raw_df = pd.DataFrame(raw_matrix, columns=raw_feature_columns)
        out_df = pd.concat([metadata.iloc[start:stop].reset_index(drop=True), raw_df], axis=1)
        if out_df.isna().any().any():
            raise ValueError(f"NaN detected while writing raw vector dataset L{length}, rows {start}:{stop}")
        out_df.to_csv(output_path, mode="w" if not wrote_header else "a", index=False, header=not wrote_header)
        wrote_header = True
        del chunk_arrays, raw_matrix, raw_df, out_df
        gc.collect()

    return expected_rows, len(METADATA_COLUMNS) + len(outcomes) + len(raw_feature_columns)


def validate_dataset(df: pd.DataFrame, outcomes: list[str], length: int, label: str) -> None:
    required = [*METADATA_COLUMNS, *outcomes]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} L{length} is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{label} L{length} has zero rows")
    if df["window_end_timestamp"].duplicated().any():
        raise ValueError(f"{label} L{length} has duplicate window_end_timestamp values")
    timestamps = pd.to_datetime(df["window_end_timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"{label} L{length} window_end_timestamp must be valid and ordered")
    if not (pd.to_numeric(df["pattern_length"], errors="coerce") == length).all():
        raise ValueError(f"{label} L{length} has an invalid pattern_length value")
    if df.isna().any().any():
        raise ValueError(f"{label} L{length} contains NaN values")


def validate_saved_csv(path: Path, outcomes: list[str], length: int, label: str) -> tuple[int, int]:
    df = pd.read_csv(path)
    validate_dataset(df, outcomes, length, label)
    return df.shape


def validate_saved_csv_streaming(
    path: Path,
    outcomes: list[str],
    length: int,
    label: str,
    expected_shape: tuple[int, int],
) -> None:
    header = pd.read_csv(path, nrows=0)
    if len(header.columns) != expected_shape[1] or len(set(header.columns)) != len(header.columns):
        raise ValueError(f"{label} L{length} has an invalid header")

    row_count = 0
    previous_timestamp: pd.Timestamp | None = None
    for chunk in pd.read_csv(path, chunksize=RAW_CHUNK_ROWS):
        validate_dataset(chunk, outcomes, length, label)
        timestamps = pd.to_datetime(chunk["window_end_timestamp"], utc=True, errors="raise")
        if previous_timestamp is not None and timestamps.iloc[0] <= previous_timestamp:
            raise ValueError(f"{label} L{length} timestamps are duplicated or out of order")
        previous_timestamp = timestamps.iloc[-1]
        row_count += len(chunk)

    if row_count != expected_shape[0]:
        raise ValueError(
            f"{label} L{length} row count mismatch: expected={expected_shape[0]} actual={row_count}"
        )


def save_dataframe_atomic(
    df: pd.DataFrame,
    output_path: Path,
    outcomes: list[str],
    length: int,
    label: str,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        prefix=f".{output_path.name}.",
        dir=output_path.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        df.to_csv(temp_path, index=False)
        shape = validate_saved_csv(temp_path, outcomes, length, f"temporary {label}")
        if shape != df.shape:
            raise ValueError(f"temporary {label} L{length} shape mismatch: expected={df.shape} actual={shape}")
        temp_path.replace(output_path)
        return shape
    finally:
        temp_path.unlink(missing_ok=True)


def process_length(
    scaled: pd.DataFrame,
    source_df: pd.DataFrame,
    feature_columns: list[str],
    outcomes: list[str],
    length: int,
) -> dict[str, object]:
    if len(source_df) < length:
        raise ValueError(f"Not enough rows for pattern_length={length}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / f"BTCUSDT_15m_similarity_summary_L{length}.csv"
    raw_path = OUTPUT_DIR / f"BTCUSDT_15m_similarity_raw_L{length}.csv"

    print(f"\n[pattern] length={length}")
    print(f"[pattern] possible windows={len(source_df) - length + 1:,}")

    summary_df = create_summary_dataset(scaled, source_df, feature_columns, outcomes, length)
    validate_dataset(summary_df, outcomes, length, "summary")
    summary_shape = save_dataframe_atomic(summary_df, summary_path, outcomes, length, "summary")
    print(f"[summary] saved: {summary_path}")
    print(f"[summary] shape={summary_df.shape}")
    del summary_df
    gc.collect()

    raw_shape = create_raw_dataset(scaled, source_df, feature_columns, outcomes, length, raw_path)
    print(f"[raw] saved: {raw_path}")
    print(f"[raw] shape={raw_shape}")

    # Each output is built as a fresh run in a same-directory temporary file,
    # validated, and atomically replaces only the completed output. Resume/append
    # of an existing dataset is intentionally not supported.
    validated_summary_shape = validate_saved_csv(summary_path, outcomes, length, "saved summary")
    print(f"[validate] summary shape={validated_summary_shape}, raw shape={raw_shape}")
    print(f"[done] pattern_length={length} completed")

    return {
        "pattern_length": length,
        "summary_path": str(summary_path),
        "summary_shape": summary_shape,
        "raw_path": str(raw_path),
        "raw_shape": raw_shape,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-resistant similarity datasets")
    parser.add_argument(
        "--scaler-fit-end",
        help="Inclusive end timestamp of the training fold used to fit scaling statistics",
    )
    parser.add_argument("--run-id", default="similarity", help="Identifier stored in scaler metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_df = load_input()
    scaled, feature_columns, excluded_features = select_and_scale_features(
        source_df,
        fit_end=args.scaler_fit_end,
        run_id=args.run_id,
    )
    outcomes = outcome_columns(source_df)
    print("[outcomes] outcome columns:")
    for column in outcomes:
        print(f"  - {column}")

    results = []
    for length in PATTERN_LENGTHS:
        results.append(process_length(scaled, source_df, feature_columns, outcomes, length))

    print("\n[complete] similarity datasets created")
    for result in results:
        print(
            f"L{result['pattern_length']}: summary_shape={result['summary_shape']} "
            f"raw_shape={result['raw_shape']}"
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, MemoryError) as error:
        print(f"[error] {error}")
        sys.exit(1)
