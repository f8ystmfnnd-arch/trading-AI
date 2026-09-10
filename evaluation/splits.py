from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitMetadata:
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    train_rows_before_purge: int
    train_rows_after_purge: int
    validation_rows_before_purge: int
    validation_rows_after_purge: int
    horizon: int


def calculate_label_end_time(
    frame: pd.DataFrame,
    horizon: int,
    *,
    timestamp_column: str = "timestamp",
) -> pd.Series:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    return timestamps.shift(-horizon)


def purged_chronological_split(
    frame: pd.DataFrame,
    horizon: int,
    *,
    timestamp_column: str = "timestamp",
    label_end_column: str | None = None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitMetadata]:
    ordered = frame.copy()
    ordered[timestamp_column] = pd.to_datetime(ordered[timestamp_column], utc=True, errors="coerce")
    if ordered[timestamp_column].isna().any():
        raise ValueError("Split timestamps contain invalid values")
    ordered = ordered.sort_values(timestamp_column).drop_duplicates(timestamp_column).reset_index(drop=True)
    row_count = len(ordered)
    train_end = int(row_count * train_fraction)
    validation_end = int(row_count * (train_fraction + validation_fraction))
    if train_end <= 0 or validation_end <= train_end or validation_end >= row_count:
        raise ValueError("Not enough rows for chronological train/validation/test splits")

    if label_end_column and label_end_column in ordered.columns:
        label_end = pd.to_datetime(ordered[label_end_column], utc=True, errors="coerce")
    else:
        label_end = calculate_label_end_time(ordered, horizon, timestamp_column=timestamp_column)

    validation_start = ordered.loc[train_end, timestamp_column]
    test_start = ordered.loc[validation_end, timestamp_column]
    train_before = ordered.iloc[:train_end].copy()
    validation_before = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()
    train = train_before.loc[label_end.iloc[:train_end] < validation_start].copy()
    validation = validation_before.loc[
        label_end.iloc[train_end:validation_end] < test_start
    ].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Purging left an empty chronological split")

    metadata = SplitMetadata(
        validation_start=validation_start,
        test_start=test_start,
        train_rows_before_purge=len(train_before),
        train_rows_after_purge=len(train),
        validation_rows_before_purge=len(validation_before),
        validation_rows_after_purge=len(validation),
        horizon=horizon,
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True), metadata
