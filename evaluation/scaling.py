from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


SCALER_VERSION = "training-fold-standardizer-v1"


@dataclass(frozen=True)
class TrainingStandardizer:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    run_id: str
    source_hash: str
    version: str = SCALER_VERSION

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"Scaler input is missing features: {missing}")
        values = frame.loc[:, self.feature_names].apply(pd.to_numeric, errors="coerce")
        return (values - np.asarray(self.mean)) / np.asarray(self.std)

    def metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "feature_order": range(len(self.feature_names)),
                "mean": self.mean,
                "std": self.std,
                "fit_start": self.fit_start.isoformat(),
                "fit_end": self.fit_end.isoformat(),
                "run_id": self.run_id,
                "source_hash": self.source_hash,
                "scaler_version": self.version,
            }
        )


def fit_training_standardizer(
    frame: pd.DataFrame,
    feature_names: list[str],
    fit_end: object,
    *,
    timestamp_column: str = "timestamp",
    run_id: str = "similarity",
) -> tuple[TrainingStandardizer, list[str]]:
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    cutoff = pd.to_datetime(fit_end, utc=True)
    fit_mask = timestamps.notna() & (timestamps <= cutoff)
    training = frame.loc[fit_mask, feature_names].apply(pd.to_numeric, errors="coerce")
    if training.empty:
        raise ValueError("Training scaler has no rows at or before fit_end")

    valid: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    excluded: list[str] = []
    for feature in feature_names:
        mean = float(training[feature].mean())
        std = float(training[feature].std(ddof=0))
        if not np.isfinite(mean) or not np.isfinite(std) or std == 0:
            excluded.append(feature)
            continue
        valid.append(feature)
        means.append(mean)
        stds.append(std)
    if not valid:
        raise ValueError("No usable features remained in the training scaler")

    hash_frame = pd.concat(
        [timestamps.loc[fit_mask].rename(timestamp_column), training.loc[:, valid]], axis=1
    )
    source_hash = hashlib.sha256(
        pd.util.hash_pandas_object(hash_frame, index=False).values.tobytes()
    ).hexdigest()
    scaler = TrainingStandardizer(
        feature_names=tuple(valid),
        mean=tuple(means),
        std=tuple(stds),
        fit_start=timestamps.loc[fit_mask].iloc[0],
        fit_end=timestamps.loc[fit_mask].iloc[-1],
        run_id=run_id,
        source_hash=source_hash,
    )
    return scaler, excluded
