from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


TARGET_POLICY_VERSION = "training-fold-threshold-v1"


@dataclass(frozen=True)
class TargetThreshold:
    value: float
    percentile: float
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    horizon: int
    target_type: str
    version: str = TARGET_POLICY_VERSION

    def metadata(self) -> dict[str, object]:
        result = asdict(self)
        result["fit_start"] = self.fit_start.isoformat()
        result["fit_end"] = self.fit_end.isoformat()
        return result


def fit_training_quantile_threshold(
    raw_target: pd.Series,
    timestamps: pd.Series,
    fit_end: object,
    *,
    label_end_time: pd.Series | None = None,
    percentile: float,
    horizon: int,
    target_type: str,
) -> TargetThreshold:
    time_values = pd.to_datetime(timestamps, utc=True, errors="coerce")
    cutoff = pd.to_datetime(fit_end, utc=True)
    raw_values = pd.to_numeric(raw_target, errors="coerce")
    mask = time_values.notna() & raw_values.notna() & (time_values <= cutoff)
    if label_end_time is not None:
        confirmed_at = pd.to_datetime(label_end_time, utc=True, errors="coerce")
        mask &= confirmed_at.notna() & (confirmed_at < cutoff)
    if not mask.any():
        raise ValueError("Target threshold has no confirmed training labels at or before fit_end")
    fitted = raw_values.loc[mask]
    return TargetThreshold(
        value=float(fitted.quantile(percentile)),
        percentile=percentile,
        fit_start=time_values.loc[mask].iloc[0],
        fit_end=time_values.loc[mask].iloc[-1],
        horizon=horizon,
        target_type=target_type,
    )


def apply_threshold(raw_target: pd.Series, threshold: TargetThreshold | float) -> pd.Series:
    value = threshold.value if isinstance(threshold, TargetThreshold) else float(threshold)
    raw_values = pd.to_numeric(raw_target, errors="coerce")
    labels = (raw_values >= value).astype("Int64")
    labels.loc[raw_values.isna()] = pd.NA
    return labels
