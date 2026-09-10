"""One monotonic risk policy shared by analysis and dashboard code."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


UNKNOWN = "UNKNOWN"
STALE = "STALE"
NORMAL = "NORMAL"
CAUTION = "CAUTION"
NO_TRADE = "NO_TRADE"


@dataclass(frozen=True)
class RiskThresholds:
    medium: float
    high: float


HIGH_VOL_THRESHOLDS = RiskThresholds(medium=0.50, high=0.70)
DROP_RISK_THRESHOLDS = RiskThresholds(medium=0.07, high=0.15)

ACTION_SEVERITY = {
    NORMAL: 0,
    CAUTION: 1,
    NO_TRADE: 2,
    STALE: 3,
    UNKNOWN: 4,
}


def _probability(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or not 0.0 <= number <= 1.0:
        return None
    return number


def risk_level(value: Any, thresholds: RiskThresholds) -> str:
    probability = _probability(value)
    if probability is None:
        return UNKNOWN
    if probability >= thresholds.high:
        return "HIGH"
    if probability >= thresholds.medium:
        return "MEDIUM"
    return "LOW"


def action_from_levels(high_vol_level: str, drop_risk_level: str) -> str:
    levels = {high_vol_level, drop_risk_level}
    if UNKNOWN in levels:
        return UNKNOWN
    if "HIGH" in levels:
        return NO_TRADE
    if "MEDIUM" in levels:
        return CAUTION
    if levels == {"LOW"}:
        return NORMAL
    return UNKNOWN


def action_from_probabilities(high_vol: Any, drop_risk: Any) -> str:
    return action_from_levels(
        risk_level(high_vol, HIGH_VOL_THRESHOLDS),
        risk_level(drop_risk, DROP_RISK_THRESHOLDS),
    )


def _utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def evaluate_snapshot(
    high_vol: Any,
    drop_risk: Any,
    *,
    feature_asof: Any = None,
    predicted_at: Any = None,
    valid_until: Any = None,
    now: Any = None,
    timestamps_match: bool = True,
) -> tuple[str, str]:
    """Validate prediction freshness before applying the shared risk policy."""
    if _probability(high_vol) is None or _probability(drop_risk) is None:
        return UNKNOWN, "risk probabilities are missing or invalid"
    if not timestamps_match:
        return UNKNOWN, "prediction timestamps or metadata do not match"

    feature_time = _utc_datetime(feature_asof)
    prediction_time = _utc_datetime(predicted_at)
    expiry_time = _utc_datetime(valid_until)
    now_time = _utc_datetime(now) if now is not None else datetime.now(timezone.utc)
    if feature_time is None or prediction_time is None or expiry_time is None or now_time is None:
        return UNKNOWN, "feature_asof, predicted_at, and valid_until are required"
    if feature_time > prediction_time or prediction_time > expiry_time or prediction_time > now_time:
        return UNKNOWN, "prediction metadata has an invalid time order"
    if now_time > expiry_time:
        return STALE, f"prediction expired at {expiry_time.isoformat()}"

    return action_from_probabilities(high_vol, drop_risk), "prediction metadata is current"


def decide_action(
    high_vol: Any,
    drop_risk: Any,
    *,
    feature_asof: Any = None,
    predicted_at: Any = None,
    valid_until: Any = None,
    now: Any = None,
    timestamps_match: bool = True,
) -> str:
    action, _ = evaluate_snapshot(
        high_vol,
        drop_risk,
        feature_asof=feature_asof,
        predicted_at=predicted_at,
        valid_until=valid_until,
        now=now,
        timestamps_match=timestamps_match,
    )
    return action
