"""Refresh an isolated spot data archive and run existing risk models (no training)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / ".runtime-packages"))

import numpy as np
import pandas as pd

from create_features_multi_timeframe import (
    MAIN_FEATURE_COLUMNS, H1_FEATURE_COLUMNS, H4_FEATURE_COLUMNS,
    add_main_15m_features, create_aux_features, merge_aux_features,
)
from create_features_with_indicators import calculate_indicators
from market.instrument import BYBIT_CATEGORY, BYBIT_SYMBOL
from risk.policy import evaluate_snapshot

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
TIMEFRAMES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def replace_with_retry(temporary: Path, path: Path) -> None:
    # Windows readers can briefly deny delete-sharing during an atomic replacement.
    for attempt in range(35):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 34:
                raise
            time.sleep(0.2)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def latest_saved_timestamp(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 4096))
        rows = handle.read().decode("utf-8").strip().splitlines()
    if not rows or rows[-1].startswith("timestamp,"):
        return None
    return pd.to_datetime(rows[-1].split(",", 1)[0], utc=True)


def atomic_append_minutes(incoming: pd.DataFrame, path: Path) -> None:
    previous_end = latest_saved_timestamp(path)
    if previous_end is None or incoming.empty:
        raise ValueError("Atomic append requires an existing non-empty archive and new rows")
    timestamps = pd.to_datetime(incoming.timestamp, utc=True)
    if timestamps.iloc[0] <= previous_end or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Incoming minutes must be unique, ordered, and newer than the archive")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copyfile(path, temporary)
        incoming[COLUMNS].to_csv(temporary, mode="a", header=False, index=False)
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def request_json(session: object, route: str, params: dict | None = None) -> dict:
    for attempt in range(4):
        try:
            url = f"https://api.bybit.com/v5/market/{route}"
            if params:
                url += "?" + urllib.parse.urlencode(params)
            with urllib.request.urlopen(url, timeout=25) as response:
                payload = json.load(response)
            if payload.get("retCode") != 0:
                raise ValueError(f"Bybit error: {payload.get('retMsg')}")
            return payload
        except (urllib.error.URLError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable retry state")


def fetch_minutes(session: object, start: pd.Timestamp, cutoff: pd.Timestamp) -> pd.DataFrame:
    cursor = int(start.timestamp() * 1000)
    stop = int(cutoff.timestamp() * 1000)
    rows = []
    batch = 0
    while cursor < stop:
        end = min(stop, cursor + 1000 * 60_000)
        payload = request_json(session, "kline", {
            "category": BYBIT_CATEGORY, "symbol": BYBIT_SYMBOL, "interval": "1",
            "start": cursor, "end": end - 1, "limit": 1000,
        })
        result = payload["result"]
        if result.get("category") != BYBIT_CATEGORY or result.get("symbol") != BYBIT_SYMBOL:
            raise ValueError("Bybit response instrument mismatch")
        rows.extend(result["list"])
        cursor = end
        batch += 1
        if batch % 10 == 0:
            print(f"[download] batches={batch}, candles={len(rows):,}", flush=True)
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["timestamp"]), unit="ms", utc=True)
    for column in COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame.loc[(frame.timestamp >= start) & (frame.timestamp < cutoff)].sort_values("timestamp").drop_duplicates("timestamp")


def complete_resample(raw: pd.DataFrame, minutes: int, cutoff: pd.Timestamp) -> pd.DataFrame:
    grouped = raw.set_index("timestamp").resample(f"{minutes}min", label="left", closed="left")
    result = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last",
                          "volume": "sum", "turnover": "sum"})
    complete = (grouped.size() == minutes) & (result.index + pd.Timedelta(minutes=minutes) <= cutoff)
    return result.loc[complete].dropna().reset_index()[COLUMNS]


def validate_recent_minutes(raw: pd.DataFrame, cutoff: pd.Timestamp, days: int = 12) -> None:
    expected = pd.date_range(cutoff - pd.Timedelta(days=days), cutoff - pd.Timedelta(minutes=1), freq="min", tz="UTC")
    recent = raw.loc[raw.timestamp >= expected[0]]
    missing = expected.difference(pd.DatetimeIndex(recent.timestamp))
    if len(missing):
        raise ValueError(f"Recent {days} days contain {len(missing)} missing minutes; first={missing[0]}")
    values = recent[COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (recent[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Recent candles contain invalid prices or missing values")
    if ((recent.high < recent[["open", "close", "low"]].max(axis=1)) |
        (recent.low > recent[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC bounds")


def build_live_features(candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    main = candles["15m"].tail(1600).copy()
    features = add_main_15m_features(main)
    for key, prefix, hours in [("1h", "h1", 1), ("4h", "h4", 4)]:
        auxiliary = create_aux_features(candles[key].tail(200), prefix, pd.Timedelta(hours=hours))
        features = merge_aux_features(features, auxiliary, key)
    names = [*MAIN_FEATURE_COLUMNS, *H1_FEATURE_COLUMNS, *H4_FEATURE_COLUMNS]
    features = features[["timestamp", *names]].merge(calculate_indicators(main), on="timestamp", validate="one_to_one")
    features = features.replace([np.inf, -np.inf], np.nan).dropna().tail(1000).reset_index(drop=True)
    features["feature_asof"] = features.timestamp + pd.Timedelta(minutes=15)
    if features.empty or features.timestamp.iloc[-1] != main.timestamp.iloc[-1]:
        raise ValueError("Latest completed candle has no valid feature vector")
    return features


def predict_existing_models(features: pd.DataFrame, model_dir: Path, now: pd.Timestamp) -> dict[str, pd.DataFrame]:
    from xgboost import Booster, DMatrix

    predictions = {}
    for kind, filename, probability_name, complement_name in [
        ("volatility", "xgb_volatility_classifier_with_indicators.json", "pred_proba_high_vol", "pred_proba_low_vol"),
        ("drop_risk", "xgb_drop_risk_classifier.json", "pred_proba_drop", "pred_proba_no_drop"),
    ]:
        path = model_dir / filename
        booster = Booster()
        booster.load_model(path)
        names = booster.feature_names
        if not names or any(name.startswith("target_") for name in names):
            raise ValueError("Model feature schema is absent or includes future targets")
        values = features.loc[:, names].astype(float)
        probabilities = booster.predict(DMatrix(values, feature_names=names))
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Model produced invalid probabilities")
        result = features[["timestamp", "feature_asof"]].copy()
        result["predicted_at"] = now
        result["valid_until"] = result.feature_asof + pd.Timedelta(minutes=15)
        result[probability_name] = probabilities
        result[complement_name] = 1 - probabilities
        result["pred_label"] = (probabilities >= 0.5).astype(int)
        result["category"] = BYBIT_CATEGORY
        result["symbol"] = BYBIT_SYMBOL
        result["model_validation_status"] = "legacy_pre_fix"
        result["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        predictions[kind] = result
    now = pd.Timestamp.now(tz="UTC")
    for result in predictions.values():
        result["predicted_at"] = now
    latest = predictions["volatility"].iloc[-1]
    action, reason = evaluate_snapshot(
        latest.pred_proba_high_vol, predictions["drop_risk"].iloc[-1].pred_proba_drop,
        feature_asof=latest.feature_asof, predicted_at=now, valid_until=latest.valid_until, now=now,
    )
    if action in {"UNKNOWN", "STALE"}:
        raise ValueError(f"Latest inference is unusable: {reason}")
    print(f"[risk] asof={latest.feature_asof}, action={action}", flush=True)
    return predictions


def refresh(raw: pd.DataFrame, data_dir: Path, model_dir: Path, session: object) -> pd.DataFrame:
    server = request_json(session, "time")
    now = pd.to_datetime(int(server["time"]), unit="ms", utc=True)
    cutoff = now.floor("min")
    start = raw.timestamp.iloc[-1] + pd.Timedelta(minutes=1)
    incoming = fetch_minutes(session, start, cutoff)
    if not incoming.empty:
        archive = data_dir / "raw" / "BTCUSDT_1m.csv"
        if archive.exists():
            atomic_append_minutes(incoming, archive)
        raw = pd.concat([raw, incoming], ignore_index=True).drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        if not archive.exists():
            atomic_csv(raw, archive)
    elif not (data_dir / "raw" / "BTCUSDT_1m.csv").exists():
        atomic_csv(raw, data_dir / "raw" / "BTCUSDT_1m.csv")
    validate_recent_minutes(raw, cutoff)
    candles = {key: complete_resample(raw, minutes, cutoff) for key, minutes in TIMEFRAMES.items()}
    for key, frame in candles.items():
        path = data_dir / "resampled" / f"BTCUSDT_{key}.csv"
        if latest_saved_timestamp(path) != frame.timestamp.iloc[-1]:
            atomic_csv(frame, path)
    features = build_live_features(candles)
    processed = data_dir / "processed"
    atomic_csv(features, processed / "BTCUSDT_15m_features_with_indicators.csv")
    now = pd.Timestamp.now(tz="UTC")
    predictions = predict_existing_models(features, model_dir, now)
    now = pd.Timestamp.now(tz="UTC")
    atomic_csv(predictions["volatility"], processed / "BTCUSDT_15m_volatility_predictions_with_indicators.csv")
    atomic_csv(predictions["drop_risk"], processed / "BTCUSDT_15m_drop_risk_predictions.csv")
    atomic_json({"status": "ok", "updated_at": now, "raw_rows": len(raw), "raw_start": raw.timestamp.iloc[0],
                 "raw_end": raw.timestamp.iloc[-1], "feature_asof": features.feature_asof.iloc[-1],
                 "category": BYBIT_CATEGORY, "symbol": BYBIT_SYMBOL, "recent_gap_check_days": 12,
                 "model_validation_status": "legacy_pre_fix"}, data_dir / "operational_status.json")
    print(f"[ready] raw={len(raw):,}, latest={raw.timestamp.iloc[-1]}", flush=True)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "operational" / "spot")
    parser.add_argument("--seed-raw", type=Path, default=Path(r"C:\Users\skana\dev\trading-AI\data\raw\BTCUSDT_1m.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path(r"C:\Users\skana\dev\trading-AI\model"))
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    # Load native inference dependencies before capturing the data cutoff.
    from xgboost import Booster, DMatrix
    archive = args.data_dir / "raw" / "BTCUSDT_1m.csv"
    source = archive if archive.exists() else args.seed_raw
    raw = pd.read_csv(source, usecols=COLUMNS)
    raw.timestamp = pd.to_datetime(raw.timestamp, utc=True, errors="raise")
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    session = None
    while True:
        try:
            raw = refresh(raw, args.data_dir, args.model_dir, session)
        except Exception as error:
            atomic_json({"status": "error", "updated_at": pd.Timestamp.now(tz="UTC"), "reason": str(error)},
                        args.data_dir / "operational_status.json")
            print(f"[error] {error}", flush=True)
            if not args.watch:
                raise
            # Reload any successfully saved archive before retrying a failed generation.
            if archive.exists():
                raw = pd.read_csv(archive, usecols=COLUMNS)
                raw.timestamp = pd.to_datetime(raw.timestamp, utc=True)
        if not args.watch:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
