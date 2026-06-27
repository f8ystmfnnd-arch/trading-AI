"""
Bybit BTC/USDT 15분봉 5년치 OHLCV 데이터 수집 스크립트.

실행 전 필요한 패키지:
    pip install ccxt pandas

실행:
    python collect_bybit_btc_15m.py
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

import ccxt
import pandas as pd


# =========================
# 수집 설정값
# =========================
SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"
MARKET_TYPE = "spot"
LOOKBACK_DAYS = 365 * 5
FETCH_LIMIT = 1000
MAX_RETRIES = 3
REQUEST_SLEEP_SECONDS = 1.5
RETRY_SLEEP_SECONDS = 5

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "data" / "btc_15m_5y.csv"

T = TypeVar("T")


def create_exchange() -> ccxt.bybit:
    """Bybit 거래소 객체를 생성합니다."""
    return ccxt.bybit(
        {
            # ccxt 자체 rate limit 기능도 켜두고, 아래 반복문에서도 추가 대기합니다.
            "enableRateLimit": True,
            "timeout": 30_000,
            "options": {
                # BTC/USDT 현물 데이터를 기준으로 수집합니다.
                # USDT 무기한 선물이 필요하면 SYMBOL을 "BTC/USDT:USDT",
                # MARKET_TYPE을 "swap"으로 바꾸면 됩니다.
                "defaultType": MARKET_TYPE,
            },
        }
    )


def run_with_retry(task: Callable[[], T], description: str) -> T | None:
    """네트워크/API 오류가 발생하면 최대 3번까지 재시도합니다."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return task()
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as error:
            if attempt == MAX_RETRIES:
                print(
                    f"[경고] {description} 실패: {MAX_RETRIES}회 재시도 후 중단합니다. "
                    f"마지막 오류: {error}"
                )
                return None

            wait_seconds = RETRY_SLEEP_SECONDS * attempt
            print(
                f"[재시도] {description} 실패 ({attempt}/{MAX_RETRIES}). "
                f"{wait_seconds}초 후 다시 요청합니다. 오류: {error}"
            )
            time.sleep(wait_seconds)

    return None


def exclusive_end_timestamp_ms(timeframe_ms: int) -> int:
    """현재 진행 중인 캔들을 제외하기 위한 종료 timestamp(ms)를 계산합니다."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return now_ms - (now_ms % timeframe_ms)


def collect_ohlcv(exchange: ccxt.bybit) -> list[list[float]]:
    """Bybit에서 5년치 15분봉 OHLCV 데이터를 순차적으로 수집합니다."""
    timeframe_ms = exchange.parse_timeframe(TIMEFRAME) * 1000
    end_ms = exclusive_end_timestamp_ms(timeframe_ms)
    since_ms = end_ms - (LOOKBACK_DAYS * 24 * 60 * 60 * 1000)
    all_rows: list[list[float]] = []

    print(
        "[시작] "
        f"{SYMBOL} {TIMEFRAME} 데이터 수집: "
        f"{datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)} ~ "
        f"{datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)}"
    )

    while since_ms < end_ms:
        readable_since = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)

        # 지정한 since 이후의 OHLCV 데이터를 한 번에 FETCH_LIMIT개까지 요청합니다.
        batch = run_with_retry(
            lambda: exchange.fetch_ohlcv(
                SYMBOL,
                timeframe=TIMEFRAME,
                since=since_ms,
                limit=FETCH_LIMIT,
            ),
            f"{readable_since} 이후 캔들 요청",
        )

        if batch is None:
            print("[중단] API 오류가 반복되어 지금까지 수집한 데이터만 저장합니다.")
            break

        if not batch:
            print("[완료] 거래소 제공 한계선까지 수집 완료")
            break

        # 마지막 완료 캔들 이후의 미완성/초과 데이터를 제외합니다.
        filtered_batch = [row for row in batch if row[0] < end_ms]
        if not filtered_batch:
            print("[완료] 마지막 완료 캔들까지 모두 수집했습니다.")
            break

        all_rows.extend(filtered_batch)

        last_timestamp = int(filtered_batch[-1][0])
        next_since_ms = last_timestamp + timeframe_ms
        if next_since_ms <= since_ms:
            print("[중단] 거래소 응답의 timestamp가 앞으로 진행되지 않아 수집을 중단합니다.")
            break
        since_ms = next_since_ms

        print(
            "[수집] "
            f"누적 {len(all_rows):,}개, "
            f"마지막 캔들: {datetime.fromtimestamp(last_timestamp / 1000, tz=timezone.utc)}"
        )

        # API 제한을 피하기 위해 각 요청 사이에 명시적으로 대기합니다.
        time.sleep(REQUEST_SLEEP_SECONDS)

    return all_rows


def save_to_csv(rows: list[list[float]]) -> None:
    """수집한 OHLCV 데이터를 DataFrame으로 변환하고 CSV 파일로 저장합니다."""
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(rows, columns=columns)

    if not df.empty:
        # 중복 캔들을 제거하고 시간순으로 정렬합니다.
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")

        # timestamp를 사람이 읽기 쉬운 UTC datetime 형식으로 변환합니다.
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[저장] {len(df):,}개 캔들을 {OUTPUT_PATH}에 저장했습니다.")


def main() -> None:
    """거래소 연결부터 데이터 수집 및 저장까지 전체 흐름을 실행합니다."""
    exchange = create_exchange()

    # 마켓 정보를 먼저 불러와 심볼이 실제로 조회 가능한지 확인합니다.
    markets = run_with_retry(exchange.load_markets, "Bybit 마켓 목록 로드")
    if markets is None:
        save_to_csv([])
        return

    if SYMBOL not in exchange.symbols:
        print(f"[오류] {SYMBOL} 심볼을 Bybit {MARKET_TYPE} 마켓에서 찾지 못했습니다.")
        save_to_csv([])
        return

    rows = collect_ohlcv(exchange)
    save_to_csv(rows)


if __name__ == "__main__":
    main()
