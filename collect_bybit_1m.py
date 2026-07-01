"""
Collect BTCUSDT 1-minute OHLCV candles from Bybit.

Examples:
    python collect_bybit_1m.py --days 365
    python collect_bybit_1m.py --update
    python collect_bybit_1m.py --days 30 --force-refresh
    python collect_bybit_1m.py --max-history --force-refresh
"""

from __future__ import annotations

import argparse
import shutil
import sys
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
MAX_HISTORY_BATCHES = 10_000
MAX_EMPTY_RESPONSES = 3

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "data" / "raw" / "BTCUSDT_1m.csv"
BACKUP_DIR = BASE_DIR / "data" / "raw" / "backups"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
ONE_MINUTE_MS = 60 * 1000


def utc_from_ms(timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)


def format_timestamp(value: pd.Timestamp | datetime | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    timestamp = pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")
    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Bybit BTCUSDT 1-minute OHLCV data into data/raw/BTCUSDT_1m.csv."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=f"Number of days to fetch from now. Default: {DAYS_TO_FETCH}.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Append only candles after the last timestamp in the existing raw CSV.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Back up the existing raw CSV, then replace it with a fresh --days fetch.",
    )
    parser.add_argument(
        "--max-history",
        action="store_true",
        help="Fetch as much BTCUSDT 1m history as Bybit provides. Requires --force-refresh.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=MAX_HISTORY_BATCHES,
        help=f"Safety limit for --max-history pagination. Default: {MAX_HISTORY_BATCHES}.",
    )
    args = parser.parse_args()

    if args.update and args.force_refresh:
        parser.error("--update and --force-refresh cannot be used together.")
    if args.update and args.days is not None:
        parser.error("--update and --days cannot be used together. Use --update by itself.")
    if args.max_history and args.update:
        parser.error("--max-history and --update cannot be used together.")
    if args.max_history and args.days is not None:
        parser.error("--max-history and --days cannot be used together.")
    if args.max_history and not args.force_refresh:
        parser.error("--max-history can replace the current raw CSV, so use it together with --force-refresh.")
    if args.days is not None and args.days <= 0:
        parser.error("--days must be a positive integer.")
    if args.max_batches <= 0:
        parser.error("--max-batches must be a positive integer.")

    args.days = DAYS_TO_FETCH if args.days is None and not args.max_history else args.days
    return args


def clean_ohlcv_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return pd.DataFrame(columns=COLUMNS), 0

    missing_columns = [column for column in COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    before_rows = len(df)
    cleaned = df[COLUMNS].copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], utc=True, errors="coerce")
    cleaned = cleaned.dropna(subset=["timestamp"])

    for column in ["open", "high", "low", "close", "volume", "turnover"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.drop_duplicates(subset="timestamp").sort_values("timestamp")
    cleaned = cleaned.reset_index(drop=True)
    removed_duplicates = before_rows - len(cleaned)
    return cleaned, removed_duplicates


def load_existing_data() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)

    df = pd.read_csv(OUTPUT_PATH)
    cleaned, removed_duplicates = clean_ohlcv_dataframe(df)
    if removed_duplicates:
        print(f"[load] duplicate/invalid rows removed from existing CSV: {removed_duplicates:,}")
    return cleaned


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

    cleaned, _ = clean_ohlcv_dataframe(df)
    return cleaned


def collect_new_data(exchange: ccxt.bybit, start_ms: int, end_ms: int) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    cursor_ms = max(0, start_ms - ONE_MINUTE_MS)

    print(
        "[start] collecting "
        f"{SYMBOL} 1m candles from {utc_from_ms(start_ms)} to {utc_from_ms(end_ms)}"
    )
    print(f"[expected] requested minute rows: {max(0, (end_ms - start_ms) // ONE_MINUTE_MS):,}")

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

        print(
            "[collect] "
            f"downloaded={sum(len(item) for item in chunks):,} "
            f"last={last_timestamp.isoformat()}"
        )

        if last_ms >= end_ms - ONE_MINUTE_MS:
            break

        if last_ms <= cursor_ms:
            next_cursor_ms = cursor_ms + ONE_MINUTE_MS
        else:
            # Use a one-candle overlap because Bybit's v5 kline boundary behavior can
            # otherwise skip one minute at each pagination boundary. Duplicates are removed later.
            next_cursor_ms = last_ms

        if next_cursor_ms <= cursor_ms:
            print("[stop] timestamp did not advance; aborting to avoid an infinite loop")
            break

        cursor_ms = next_cursor_ms
        time.sleep(REQUEST_SLEEP_SECONDS)

    if not chunks:
        return pd.DataFrame(columns=COLUMNS)

    combined, _ = clean_ohlcv_dataframe(pd.concat(chunks, ignore_index=True))
    combined = combined[combined["timestamp"] >= pd.Timestamp(utc_from_ms(start_ms))]
    combined = combined.reset_index(drop=True)
    return combined



def collect_max_history(exchange: ccxt.bybit, end_ms: int, max_batches: int) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    cursor_end_ms = end_ms
    previous_earliest_ms: int | None = None
    empty_responses = 0

    print("[mode] max-history")
    print(f"[start] collecting {SYMBOL} 1m candles backward from {utc_from_ms(end_ms)}")
    print(f"[safety] max_batches={max_batches:,}, max_empty_responses={MAX_EMPTY_RESPONSES}")

    try:
        for batch_number in range(1, max_batches + 1):
            batch_start_ms = max(0, cursor_end_ms - FETCH_LIMIT * ONE_MINUTE_MS)
            rows = request_kline(exchange, batch_start_ms, cursor_end_ms)
            chunk = normalize_rows(rows, cursor_end_ms + ONE_MINUTE_MS)
            chunk = chunk[chunk["timestamp"] < pd.Timestamp(utc_from_ms(cursor_end_ms))]

            if chunk.empty:
                empty_responses += 1
                print(
                    f"[max-history] batch={batch_number:,} rows=0 "
                    f"empty_responses={empty_responses}/{MAX_EMPTY_RESPONSES}"
                )
                if empty_responses >= MAX_EMPTY_RESPONSES:
                    print("[stop] repeated empty responses; reached the available historical boundary")
                    break
                cursor_end_ms = batch_start_ms
                time.sleep(REQUEST_SLEEP_SECONDS)
                continue

            empty_responses = 0
            chunks.append(chunk)
            earliest = chunk["timestamp"].iloc[0]
            latest = chunk["timestamp"].iloc[-1]
            earliest_ms = int(earliest.timestamp() * 1000)

            combined_so_far, _ = clean_ohlcv_dataframe(pd.concat(chunks, ignore_index=True))
            print(
                f"[max-history] batch={batch_number:,} rows={len(chunk):,} "
                f"total={len(combined_so_far):,} "
                f"earliest={combined_so_far['timestamp'].iloc[0].isoformat()} "
                f"latest={combined_so_far['timestamp'].iloc[-1].isoformat()}"
            )

            if previous_earliest_ms is not None and earliest_ms >= previous_earliest_ms:
                print("[stop] earliest timestamp did not move backward; stopping to avoid a loop")
                break

            previous_earliest_ms = earliest_ms
            if earliest_ms <= ONE_MINUTE_MS:
                print("[stop] reached timestamp near Unix epoch; stopping")
                break

            # Keep a one-candle overlap while moving backward. Duplicates are removed before saving.
            cursor_end_ms = earliest_ms + ONE_MINUTE_MS
            time.sleep(REQUEST_SLEEP_SECONDS)
        else:
            print("[stop] max_batches safety limit reached")
    except KeyboardInterrupt:
        print("\n[interrupt] KeyboardInterrupt received; saving collected rows so far")

    if not chunks:
        return pd.DataFrame(columns=COLUMNS)

    combined, _ = clean_ohlcv_dataframe(pd.concat(chunks, ignore_index=True))
    return combined

def backup_existing_raw() -> Path | None:
    if not OUTPUT_PATH.exists():
        return None

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"BTCUSDT_1m_{stamp}.csv"
    shutil.copy2(OUTPUT_PATH, backup_path)
    print(f"[backup] existing raw CSV backed up to: {backup_path}")
    return backup_path


def combine_and_save(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before_rows = len(existing_df) + len(new_df)
    combined, removed_duplicates = clean_ohlcv_dataframe(pd.concat([existing_df, new_df], ignore_index=True))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"[save] before clean rows={before_rows:,}")
    print(f"[save] duplicate timestamps removed: {removed_duplicates:,}")
    print(f"[save] rows={len(combined):,} path={OUTPUT_PATH}")
    return combined, removed_duplicates


def print_dataset_summary(df: pd.DataFrame, removed_duplicates: int = 0, added_rows: int | None = None) -> None:
    print("\n[data summary]")
    print(f"Raw 1m rows: {len(df):,}")
    if not df.empty:
        print(f"Start: {format_timestamp(df['timestamp'].iloc[0])}")
        print(f"End: {format_timestamp(df['timestamp'].iloc[-1])}")
    print(f"Duplicate timestamps removed: {removed_duplicates:,}")
    if added_rows is not None:
        print(f"New rows added: {added_rows:,}")
    print_gap_summary(df)


def print_gap_summary(df: pd.DataFrame, preview_count: int = 10) -> None:
    if df.empty or len(df) < 2:
        print("1m interval gaps: 0")
        print("Max gap: -")
        return

    diffs = df["timestamp"].diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(minutes=1)]
    print(f"1m interval gaps: {len(gaps):,}")
    print(f"Max gap: {gaps.max() if not gaps.empty else pd.Timedelta(minutes=1)}")

    if not gaps.empty:
        print("Gap examples:")
        for index, gap in gaps.head(preview_count).items():
            previous_timestamp = df.loc[index - 1, "timestamp"]
            current_timestamp = df.loc[index, "timestamp"]
            print(
                f"  {format_timestamp(previous_timestamp)} -> "
                f"{format_timestamp(current_timestamp)} | gap={gap}"
            )


def run_fresh_fetch(days: int, force_refresh: bool) -> None:
    if OUTPUT_PATH.exists() and not force_refresh:
        print(f"[warning] raw CSV already exists: {OUTPUT_PATH}")
        print("[warning] Not overwriting existing data. Use --update to append new candles.")
        print("[warning] Use --force-refresh with --days if you want a fresh fetch with backup.")
        existing_df = load_existing_data()
        print_dataset_summary(existing_df)
        return

    if force_refresh:
        backup_existing_raw()

    end_ms = completed_minute_ms()
    start_ms = end_ms - days * 24 * 60 * ONE_MINUTE_MS
    print(f"[mode] fresh fetch days={days}")
    print(f"[period] start={utc_from_ms(start_ms)} end={utc_from_ms(end_ms)}")

    exchange = create_exchange()
    new_df = collect_new_data(exchange, start_ms, end_ms)
    saved_df, removed_duplicates = combine_and_save(pd.DataFrame(columns=COLUMNS), new_df)
    print_dataset_summary(saved_df, removed_duplicates=removed_duplicates, added_rows=len(saved_df))


def run_max_history(force_refresh: bool, max_batches: int) -> None:
    if not force_refresh:
        raise ValueError("--max-history can replace the current raw CSV, so use it together with --force-refresh.")

    backup_path = backup_existing_raw()
    end_ms = completed_minute_ms()
    exchange = create_exchange()
    new_df = collect_max_history(exchange, end_ms=end_ms, max_batches=max_batches)
    saved_df, removed_duplicates = combine_and_save(pd.DataFrame(columns=COLUMNS), new_df)
    print_dataset_summary(saved_df, removed_duplicates=removed_duplicates, added_rows=len(saved_df))
    print("[mode] requested mode: max-history")
    print(f"[save] path={OUTPUT_PATH}")
    print(f"[backup] path={backup_path if backup_path else '-'}")


def run_update() -> None:
    if not OUTPUT_PATH.exists():
        print("[update] existing raw CSV was not found.")
        print("[update] Please run: python collect_bybit_1m.py --days 365")
        return

    existing_df = load_existing_data()
    if existing_df.empty:
        print("[update] existing raw CSV is empty.")
        print("[update] Please run: python collect_bybit_1m.py --days 365 --force-refresh")
        return

    last_timestamp = existing_df["timestamp"].iloc[-1]
    start_ms = int(last_timestamp.timestamp() * 1000) + ONE_MINUTE_MS
    end_ms = completed_minute_ms()

    print(f"[mode] update existing raw CSV")
    print(f"[resume] existing rows={len(existing_df):,}; last={last_timestamp.isoformat()}")

    if start_ms >= end_ms:
        print("[done] existing file is already up to date; no new completed 1m candle to fetch.")
        print_dataset_summary(existing_df, added_rows=0)
        return

    exchange = create_exchange()
    new_df = collect_new_data(exchange, start_ms, end_ms)
    before_existing_rows = len(existing_df)
    saved_df, removed_duplicates = combine_and_save(existing_df, new_df)
    added_rows = max(0, len(saved_df) - before_existing_rows)
    print_dataset_summary(saved_df, removed_duplicates=removed_duplicates, added_rows=added_rows)


def main() -> None:
    args = parse_args()
    try:
        if args.update:
            run_update()
        elif args.max_history:
            run_max_history(force_refresh=args.force_refresh, max_batches=args.max_batches)
        else:
            run_fresh_fetch(days=args.days, force_refresh=args.force_refresh)
    except (RuntimeError, ValueError, OSError) as error:
        print(f"[error] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
