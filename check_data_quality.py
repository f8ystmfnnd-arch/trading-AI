"""
Check raw and resampled BTCUSDT OHLCV CSV files.

Run:
    python check_data_quality.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASETS = {
    "1m raw": (BASE_DIR / "data" / "raw" / "BTCUSDT_1m.csv", pd.Timedelta(minutes=1)),
    "5m": (BASE_DIR / "data" / "resampled" / "BTCUSDT_5m.csv", pd.Timedelta(minutes=5)),
    "15m": (BASE_DIR / "data" / "resampled" / "BTCUSDT_15m.csv", pd.Timedelta(minutes=15)),
    "1h": (BASE_DIR / "data" / "resampled" / "BTCUSDT_1h.csv", pd.Timedelta(hours=1)),
    "4h": (BASE_DIR / "data" / "resampled" / "BTCUSDT_4h.csv", pd.Timedelta(hours=4)),
    "1d": (BASE_DIR / "data" / "resampled" / "BTCUSDT_1d.csv", pd.Timedelta(days=1)),
}

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def format_timestamp(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return value.isoformat()


def analyze_dataset(name: str, path: Path, expected_interval: pd.Timedelta) -> None:
    print("=" * 78)
    print(f"{name} | {path}")

    if not path.exists():
        print("status: missing")
        return

    df = load_dataset(path)
    row_count = len(df)
    duplicate_count = int(df["timestamp"].duplicated().sum())
    missing_value_count = int(df.isna().sum().sum())

    clean = df.dropna(subset=["timestamp"]).drop_duplicates(subset="timestamp")
    clean = clean.sort_values("timestamp").reset_index(drop=True)

    start = clean["timestamp"].iloc[0] if not clean.empty else None
    end = clean["timestamp"].iloc[-1] if not clean.empty else None
    diffs = clean["timestamp"].diff().dropna()
    bad_diffs = diffs[diffs != expected_interval]

    print(f"rows: {row_count:,}")
    print(f"start: {format_timestamp(start)}")
    print(f"end: {format_timestamp(end)}")
    print(f"duplicate timestamps: {duplicate_count:,}")
    print(f"missing values: {missing_value_count:,}")
    print(f"expected interval: {expected_interval}")
    print(f"interval issues: {len(bad_diffs):,}")

    if not bad_diffs.empty:
        preview = bad_diffs.head(10)
        print("interval issue preview:")
        for index, actual_interval in preview.items():
            previous_timestamp = clean.loc[index - 1, "timestamp"]
            current_timestamp = clean.loc[index, "timestamp"]
            print(
                f"  {previous_timestamp.isoformat()} -> "
                f"{current_timestamp.isoformat()} | actual={actual_interval}"
            )


def main() -> None:
    print("BTCUSDT data quality report")
    for name, (path, expected_interval) in DATASETS.items():
        analyze_dataset(name, path, expected_interval)


if __name__ == "__main__":
    main()
