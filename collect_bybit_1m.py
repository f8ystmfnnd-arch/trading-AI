"""
Collect BTCUSDT 1-minute OHLCV candles from Bybit.

Default output:
    data/raw/BTCUSDT_1m.csv

Run:
    python collect_bybit_1m.py
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd


SYMBOL = "BTCUSDT"
CATEGORY = "spot"
INTERVAL = "1"
DAYS_TO_FETCH = 365
FETCH_LIMIT = 1000
MAX_RETRIES = 5
REQUEST_SLEEP_SECONDS = 0.35
RETRY_SLEEP_SECONDS = 5

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "data" / "raw" / "BTCUSDT_1m.csv"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
ONE_MINUTE_MS = 60 * 1000


def utc_from_ms(timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)


def completed_minute_ms() -> int:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return now_ms - (now_ms % ONE_MINUTE_MS)


def create_exchange() -> ccxt.bybit:
    return ccxt.bybit(
        {
            "enableRateLimit": True,
            "timeout": 30_000,
            "options": {"defaultType": CATEGORY},
        }
    )


def load_existing_data() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)

    df = pd.read_csv(OUTPUT_PATH)
    missing_columns = [column for column in COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Existing file is missing required columns: {missing_columns}")

    df = df[COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    for column in ["open", "high", "low", "close", "volume", "turnover"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def request_kline(exchange: ccxt.bybit, start_ms: int, end_ms: int) -> list[list[Any]]:
    params = {
        "category": CATEGORY,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "start": start_ms,
        "end": end_ms,
        "limit": FETCH_LIMIT,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = exchange.publicGetV5MarketKline(params)
            result = response.get("result", {})
            rows = result.get("list", [])
            if not isinstance(rows, list):
                raise ValueError(f"Unexpected Bybit response shape: {response}")
            return rows
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Bybit request failed after {MAX_RETRIES} attempts: {error}"
                ) from error

            wait_seconds = RETRY_SLEEP_SECONDS * attempt
            print(
                f"[retry] request failed ({attempt}/{MAX_RETRIES}); "
                f"sleeping {wait_seconds}s. error={error}"
            )
            time.sleep(wait_seconds)

    return []


def normalize_rows(rows: list[list[Any]], end_ms: int) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if len(row) < 7:
            continue

        timestamp_ms = int(row[0])
        if timestamp_ms >= end_ms:
            continue

        normalized.append(
            {
                "timestamp": utc_from_ms(timestamp_ms),
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "turnover": row[6],
            }
        )

    df = pd.DataFrame(normalized, columns=COLUMNS)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for column in ["open", "high", "low", "close", "volume", "turnover"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.drop_duplicates(subset="timestamp").sort_values("timestamp")


def collect_new_data(exchange: ccxt.bybit, start_ms: int, end_ms: int) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    cursor_ms = start_ms

    print(
        "[start] collecting "
        f"{SYMBOL} 1m candles from {utc_from_ms(start_ms)} to {utc_from_ms(end_ms)}"
    )

    while cursor_ms < end_ms:
        batch_end_ms = min(end_ms, cursor_ms + FETCH_LIMIT * ONE_MINUTE_MS)
        rows = request_kline(exchange, cursor_ms, batch_end_ms)
        chunk = normalize_rows(rows, end_ms)

        if chunk.empty:
            print(f"[stop] no candles returned after {utc_from_ms(cursor_ms)}")
            break

        chunks.append(chunk)
        last_timestamp = chunk["timestamp"].iloc[-1]
        last_ms = int(last_timestamp.timestamp() * 1000)
        next_cursor_ms = last_ms + ONE_MINUTE_MS

        print(
            "[collect] "
            f"new={sum(len(item) for item in chunks):,} "
            f"last={last_timestamp.isoformat()}"
        )

        if next_cursor_ms <= cursor_ms:
            print("[stop] timestamp did not advance; aborting to avoid an infinite loop")
            break

        cursor_ms = next_cursor_ms
        time.sleep(REQUEST_SLEEP_SECONDS)

    if not chunks:
        return pd.DataFrame(columns=COLUMNS)

    return pd.concat(chunks, ignore_index=True)


def save_data(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    if not combined.empty:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
        combined = combined.dropna(subset=["timestamp"])
        combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")
        combined = combined.reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"[save] rows={len(combined):,} path={OUTPUT_PATH}")


def main() -> None:
    existing_df = load_existing_data()
    end_ms = completed_minute_ms()

    if existing_df.empty:
        start_ms = end_ms - DAYS_TO_FETCH * 24 * 60 * ONE_MINUTE_MS
        print(f"[resume] no existing file; fetching the latest {DAYS_TO_FETCH} days")
    else:
        last_timestamp = existing_df["timestamp"].iloc[-1]
        start_ms = int(last_timestamp.timestamp() * 1000) + ONE_MINUTE_MS
        print(f"[resume] existing rows={len(existing_df):,}; last={last_timestamp.isoformat()}")

    if start_ms >= end_ms:
        print("[done] existing file is already up to date")
        save_data(existing_df, pd.DataFrame(columns=COLUMNS))
        return

    exchange = create_exchange()
    new_df = collect_new_data(exchange, start_ms, end_ms)
    save_data(existing_df, new_df)


if __name__ == "__main__":
    main()
