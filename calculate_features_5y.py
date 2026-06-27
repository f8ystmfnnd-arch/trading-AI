"""
Bybit BTC/USDT 15분봉 5년치 데이터에서 머신러닝 학습용 피처와 타겟을 생성합니다.

입력:
    data/btc_15m_5y.csv

출력:
    data/btc_15m_5y_features.csv

실행:
    python calculate_features_5y.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 파일 경로 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "btc_15m_5y.csv"
OUTPUT_PATH = BASE_DIR / "data" / "btc_15m_5y_features.csv"


# =========================
# 15분봉 기준 지표 설정
# =========================
CANDLES_PER_DAY = 96
SMA_5D_WINDOW = CANDLES_PER_DAY * 5
SMA_20D_WINDOW = CANDLES_PER_DAY * 20
SMA_60D_WINDOW = CANDLES_PER_DAY * 60

RSI_WINDOW = 14
BOLLINGER_WINDOW = 20
BOLLINGER_STD_MULTIPLIER = 2
VOLUME_AVG_WINDOW = 20
TARGET_FORWARD_CANDLES = 16

FEATURE_COLUMNS = [
    "close_sma_5d_ratio",
    "close_sma_20d_ratio",
    "close_sma_60d_ratio",
    "rsi_14",
    "bb_bandwidth_20",
    "volume_ratio_20",
]

OUTPUT_COLUMNS = ["timestamp", *FEATURE_COLUMNS, "target"]


def load_price_data() -> pd.DataFrame:
    """원본 15분봉 CSV를 읽고 시간순 정렬 및 숫자형 변환을 수행합니다."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}")

    print(f"[로드] 원본 15분봉 데이터를 읽습니다: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"입력 파일에 필수 컬럼이 없습니다: {missing_columns}")

    # timestamp는 UTC 기준 datetime으로 파싱하고, 캔들 순서가 섞였을 가능성에 대비해 정렬합니다.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    # 지표 계산에 쓰는 가격/거래량 컬럼은 숫자형으로 강제 변환합니다.
    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    print(f"[로드 완료] 정렬 및 중복 제거 후 캔들 수: {len(df):,}개")
    return df


def add_sma_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """현재 종가를 5일, 20일, 60일 SMA로 나눈 이동평균선 이격도 피처를 생성합니다."""
    close = df["close"]

    # 절대 가격을 직접 쓰지 않고, 현재 종가가 각 이동평균 대비 몇 배인지 비율로 표현합니다.
    df["close_sma_5d_ratio"] = close / close.rolling(SMA_5D_WINDOW).mean()
    df["close_sma_20d_ratio"] = close / close.rolling(SMA_20D_WINDOW).mean()
    df["close_sma_60d_ratio"] = close / close.rolling(SMA_60D_WINDOW).mean()

    print(
        "[피처] 이동평균선 이격도 생성 완료 "
        f"(5일={SMA_5D_WINDOW}캔들, 20일={SMA_20D_WINDOW}캔들, 60일={SMA_60D_WINDOW}캔들)"
    )
    return df


def add_rsi_feature(df: pd.DataFrame) -> pd.DataFrame:
    """14캔들 기준 RSI 피처를 생성합니다."""
    close_diff = df["close"].diff()

    # 상승분과 하락분을 분리해 14캔들 평균 상승폭/하락폭을 계산합니다.
    gain = close_diff.clip(lower=0)
    loss = (-close_diff).clip(lower=0)
    avg_gain = gain.rolling(RSI_WINDOW).mean()
    avg_loss = loss.rolling(RSI_WINDOW).mean()

    # 평균 하락폭이 0인 구간은 0으로 나누지 않도록 별도 처리합니다.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)

    df["rsi_14"] = rsi
    print(f"[피처] RSI 생성 완료 ({RSI_WINDOW}캔들 기준)")
    return df


def add_bollinger_bandwidth_feature(df: pd.DataFrame) -> pd.DataFrame:
    """20캔들 기준 볼린저 밴드 너비 피처를 생성합니다."""
    close = df["close"]

    # 일반적인 볼린저 밴드 설정인 20캔들 이동평균과 2표준편차를 사용합니다.
    middle_band = close.rolling(BOLLINGER_WINDOW).mean()
    band_std = close.rolling(BOLLINGER_WINDOW).std(ddof=0)
    upper_band = middle_band + (BOLLINGER_STD_MULTIPLIER * band_std)
    lower_band = middle_band - (BOLLINGER_STD_MULTIPLIER * band_std)

    # 밴드 폭을 중간 밴드로 나눠 변동성을 절대 가격이 아닌 상대 크기로 표현합니다.
    df["bb_bandwidth_20"] = (upper_band - lower_band) / middle_band

    print(
        "[피처] 볼린저 밴드 너비 생성 완료 "
        f"({BOLLINGER_WINDOW}캔들, {BOLLINGER_STD_MULTIPLIER}표준편차 기준)"
    )
    return df


def add_volume_ratio_feature(df: pd.DataFrame) -> pd.DataFrame:
    """현재 거래량을 최근 20캔들 평균 거래량으로 나눈 거래량 비율 피처를 생성합니다."""
    volume_avg = df["volume"].rolling(VOLUME_AVG_WINDOW).mean()

    # 절대 거래량 대신 최근 평균 대비 현재 거래량의 상대적 크기를 사용합니다.
    df["volume_ratio_20"] = df["volume"] / volume_avg

    print(f"[피처] 거래량 비율 생성 완료 ({VOLUME_AVG_WINDOW}캔들 평균 기준)")
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """현재 종가 대비 16캔들 뒤 종가의 4시간 미래 수익률 타겟을 생성합니다."""
    future_close = df["close"].shift(-TARGET_FORWARD_CANDLES)

    # target은 미래 종가 수익률이며, 마지막 16개 행은 미래 가격이 없어 NaN이 됩니다.
    df["target"] = (future_close / df["close"]) - 1

    print(f"[타겟] 4시간 뒤 수익률 target 생성 완료 ({TARGET_FORWARD_CANDLES}캔들 뒤)")
    return df


def clean_and_select_output(df: pd.DataFrame) -> pd.DataFrame:
    """결측치와 무한대를 제거하고 학습에 필요한 상대 피처와 타겟만 선택합니다."""
    before_rows = len(df)

    # 0으로 나누기 등으로 생길 수 있는 무한대 값을 NaN으로 바꾼 뒤 한 번에 제거합니다.
    output_df = df[OUTPUT_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    dropped_rows = before_rows - len(output_df)
    print(f"[정리] NaN/무한대 행 제거 완료: {dropped_rows:,}개 제거, 최종 {len(output_df):,}개 행")
    return output_df


def save_features(df: pd.DataFrame) -> None:
    """최종 피처 데이터프레임을 CSV로 저장합니다."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[저장] 학습용 피처 파일 저장 완료: {OUTPUT_PATH}")


def print_latest_preview(df: pd.DataFrame) -> None:
    """터미널에서 최신 5개 행의 주요 피처와 타겟을 보기 좋게 출력합니다."""
    preview_columns = [
        "timestamp",
        "close_sma_5d_ratio",
        "close_sma_20d_ratio",
        "close_sma_60d_ratio",
        "rsi_14",
        "bb_bandwidth_20",
        "volume_ratio_20",
        "target",
    ]

    print("\n[미리보기] 최신 5개 행의 주요 피처와 타겟")
    with pd.option_context("display.width", 180, "display.max_columns", None, "display.float_format", "{:.8f}".format):
        print(df[preview_columns].tail(5).to_string(index=False))


def main() -> None:
    """원본 데이터 로드부터 피처/타겟 생성, 저장, 미리보기 출력까지 전체 과정을 실행합니다."""
    df = load_price_data()
    df = add_sma_ratio_features(df)
    df = add_rsi_feature(df)
    df = add_bollinger_bandwidth_feature(df)
    df = add_volume_ratio_feature(df)
    df = add_target(df)

    output_df = clean_and_select_output(df)
    save_features(output_df)
    print_latest_preview(output_df)


if __name__ == "__main__":
    main()
