"""
학습된 XGBoost 모델의 최근 20% Test 구간 선물 양방향 백테스트 스크립트.

핵심 아이디어:
    - Buy & Hold: BTC를 계속 보유했을 때의 기준 성과
    - AI 선물 양방향 전략: 모델 예측값에 따라 Long / Short / Cash 상태를 관리
    - 리스크 관리: 익절, 손절, 쿨다운, 선물 수수료를 모두 반영

입력:
    data/btc_15m_5y_features.csv
    data/btc_15m_5y.csv
    model/xgb_model.json

출력:
    model/backtest_result.png

실행:
    python run_backtest.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


# Matplotlib은 최초 실행 시 폰트 캐시를 생성합니다.
# 권한 문제를 피하기 위해 프로젝트 내부의 쓰기 가능한 캐시 폴더를 사용합니다.
BASE_DIR = Path(__file__).resolve().parent
MPL_CONFIG_DIR = BASE_DIR / ".matplotlib_cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

try:
    import matplotlib.pyplot as plt
except ImportError as error:
    raise ImportError("matplotlib이 설치되어 있지 않습니다. `pip install -r requirements.txt`를 실행해주세요.") from error

try:
    from xgboost import XGBRegressor
except ImportError as error:
    raise ImportError("xgboost가 설치되어 있지 않습니다. `pip install -r requirements.txt`를 실행해주세요.") from error


# =========================
# 파일 경로 및 백테스트 설정
# =========================
FEATURE_PATH = BASE_DIR / "data" / "btc_15m_5y_features.csv"
RAW_PRICE_PATH = BASE_DIR / "data" / "btc_15m_5y.csv"
MODEL_PATH = BASE_DIR / "model" / "xgb_model.json"
BACKTEST_IMAGE_PATH = BASE_DIR / "model" / "backtest_result.png"

TRAIN_RATIO = 0.8

# 선물 시장가 거래 수수료를 0.05%로 가정합니다.
FEE_RATE = 0.0005

# 모델 예측값이 충분히 강할 때만 신규 포지션을 엽니다.
ENTRY_LONG_THRESHOLD = 0.005
ENTRY_SHORT_THRESHOLD = -0.005

# 포지션 보유 중 예측이 확실히 반대로 꺾였을 때만 일반 청산합니다.
EXIT_LONG_THRESHOLD = -0.001
EXIT_SHORT_THRESHOLD = 0.001

# 진입가 대비 익절/손절 기준입니다.
TAKE_PROFIT_PCT = 0.015
STOP_LOSS_PCT = 0.008

# 익절/손절 직후 과도한 재진입을 막기 위한 1시간 쿨다운입니다.
COOLDOWN_CANDLES = 4

# 모델 학습 때와 동일하게 timestamp, target, 원본 OHLCV 절대값은 피처에서 제외합니다.
EXCLUDED_FEATURE_COLUMNS = {"timestamp", "target", "open", "high", "low", "close", "volume"}


def load_feature_data() -> pd.DataFrame:
    """학습용 피처 데이터를 불러오고 시간 순서대로 정렬합니다."""
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"피처 파일을 찾을 수 없습니다: {FEATURE_PATH}")

    print(f"[로드] 피처 데이터를 불러옵니다: {FEATURE_PATH}")
    df = pd.read_csv(FEATURE_PATH)

    if "timestamp" not in df.columns:
        raise ValueError("피처 파일에 timestamp 컬럼이 없습니다.")
    if "target" not in df.columns:
        raise ValueError("피처 파일에 target 컬럼이 없습니다.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    print(f"[로드 완료] 피처 데이터 행 수: {len(df):,}개")
    return df


def load_raw_15m_market_data() -> pd.DataFrame:
    """원본 OHLCV에서 다음 15분 수익률과 다음 캔들의 고가/저가를 계산합니다."""
    if not RAW_PRICE_PATH.exists():
        raise FileNotFoundError(f"원본 가격 파일을 찾을 수 없습니다: {RAW_PRICE_PATH}")

    print(f"[로드] 리스크 관리용 OHLCV 데이터를 불러옵니다: {RAW_PRICE_PATH}")
    price_df = pd.read_csv(RAW_PRICE_PATH, usecols=["timestamp", "high", "low", "close"])
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)

    for column in ["high", "low", "close"]:
        price_df[column] = pd.to_numeric(price_df[column], errors="coerce")

    price_df = price_df.sort_values("timestamp").drop_duplicates(subset="timestamp").dropna().reset_index(drop=True)

    # 모델 신호는 현재 캔들 종가 기준으로 생성된다고 가정합니다.
    # 따라서 실제 손익 및 익절/손절 도달 여부는 다음 15분 캔들의 OHLC로 평가합니다.
    price_df["next_close"] = price_df["close"].shift(-1)
    price_df["next_high"] = price_df["high"].shift(-1)
    price_df["next_low"] = price_df["low"].shift(-1)
    price_df["btc_15m_return"] = price_df["next_close"] / price_df["close"] - 1

    market_df = price_df[
        ["timestamp", "close", "next_high", "next_low", "next_close", "btc_15m_return"]
    ].dropna().reset_index(drop=True)

    print(f"[로드 완료] 실제 15분 시장 데이터 행 수: {len(market_df):,}개")
    return market_df


def select_test_set(df: pd.DataFrame) -> pd.DataFrame:
    """학습과 동일하게 앞 80%를 제외하고 마지막 20%만 Test Set으로 선택합니다."""
    split_index = int(len(df) * TRAIN_RATIO)
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Test Set을 만들기에 데이터가 충분하지 않습니다.")

    test_df = df.iloc[split_index:].reset_index(drop=True)

    print(f"[분할] 마지막 20% Test Set 선택 완료: {len(test_df):,}개")
    print(f"[분할] Test 기간: {test_df['timestamp'].iloc[0]} ~ {test_df['timestamp'].iloc[-1]}")
    return test_df


def load_model() -> XGBRegressor:
    """저장된 XGBoost 회귀 모델을 불러옵니다."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))
    print(f"[모델] 저장된 XGBoost 모델을 불러왔습니다: {MODEL_PATH}")
    return model


def predict_test_returns(model: XGBRegressor, test_df: pd.DataFrame) -> pd.DataFrame:
    """Test Set에 대해 4시간 뒤 수익률 예측값 y_pred를 생성합니다."""
    feature_columns = [column for column in test_df.columns if column not in EXCLUDED_FEATURE_COLUMNS]
    if not feature_columns:
        raise ValueError("예측에 사용할 피처 컬럼이 없습니다.")

    test_df = test_df.copy()
    test_df["y_pred"] = model.predict(test_df[feature_columns])

    print(f"[예측] Test Set 예측 완료. 사용 피처: {feature_columns}")
    print(f"[예측] 롱 진입 기준 초과 비율: {(test_df['y_pred'] > ENTRY_LONG_THRESHOLD).mean() * 100:.2f}%")
    print(f"[예측] 숏 진입 기준 미만 비율: {(test_df['y_pred'] < ENTRY_SHORT_THRESHOLD).mean() * 100:.2f}%")
    return test_df


def calculate_mdd(equity_curve: pd.Series) -> float:
    """누적 수익률 곡선에서 최대 낙폭(MDD)을 계산합니다."""
    # 지금까지의 최고 자산가치(running peak)를 계속 기록합니다.
    running_peak = equity_curve.cummax()

    # 현재 자산가치가 이전 최고점 대비 얼마나 하락했는지를 drawdown으로 계산합니다.
    # 예: equity=0.8, peak=1.0이면 drawdown=-20%입니다.
    drawdown = equity_curve / running_peak - 1

    # MDD는 전체 기간 중 가장 깊었던 하락폭입니다.
    return float(drawdown.min())


def calculate_zero_threshold_trade_count(y_pred: pd.Series) -> int:
    """이전 0 기준 단순 Long/Cash 로직의 포지션 변경 횟수를 참고용으로 계산합니다."""
    legacy_position = pd.Series(np.where(y_pred > 0, 1, 0), index=y_pred.index)
    legacy_previous_position = legacy_position.shift(1).fillna(0)
    legacy_position_change = (legacy_position - legacy_previous_position).abs()
    return int(legacy_position_change.sum())


def position_to_number(position: str) -> int:
    """문자열 포지션을 수익률 계산용 숫자 포지션으로 변환합니다."""
    if position == "Long":
        return 1
    if position == "Short":
        return -1
    return 0


def build_futures_positions(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV를 캔들별로 순회하며 Long/Short/Cash 포지션과 전략 수익률을 계산합니다."""
    current_position = "Cash"
    entry_price: float | None = None
    cooldown_remaining = 0

    records: list[dict[str, float | int | str]] = []

    for row in backtest_df.itertuples(index=False):
        y_pred = float(row.y_pred)
        close_price = float(row.close)
        next_high = float(row.next_high)
        next_low = float(row.next_low)
        next_close = float(row.next_close)

        starting_position = current_position
        fee_events = 0
        position_change_events = 0
        long_entry = 0
        short_entry = 0
        take_profit_exit = 0
        stop_loss_exit = 0
        threshold_exit = 0
        cooldown_blocked = 0
        action_reason = "Hold"

        # 1) 먼저 모델 예측 변화로 인한 청산/반전 여부를 판단합니다.
        if current_position == "Long":
            if y_pred < ENTRY_SHORT_THRESHOLD:
                # 강한 하락 예측이면 Long을 닫고 Short으로 바로 전환합니다.
                current_position = "Short"
                entry_price = close_price
                fee_events += 2
                position_change_events += 1
                short_entry = 1
                threshold_exit = 1
                action_reason = "Reverse Long to Short"
            elif y_pred < EXIT_LONG_THRESHOLD:
                # 약한 반대 신호는 바로 숏 진입하지 않고 Cash로 일반 청산합니다.
                current_position = "Cash"
                entry_price = None
                fee_events += 1
                position_change_events += 1
                threshold_exit = 1
                action_reason = "Threshold Exit Long"

        elif current_position == "Short":
            if y_pred > ENTRY_LONG_THRESHOLD:
                # 강한 상승 예측이면 Short을 닫고 Long으로 바로 전환합니다.
                current_position = "Long"
                entry_price = close_price
                fee_events += 2
                position_change_events += 1
                long_entry = 1
                threshold_exit = 1
                action_reason = "Reverse Short to Long"
            elif y_pred > EXIT_SHORT_THRESHOLD:
                # 약한 반대 신호는 바로 롱 진입하지 않고 Cash로 일반 청산합니다.
                current_position = "Cash"
                entry_price = None
                fee_events += 1
                position_change_events += 1
                threshold_exit = 1
                action_reason = "Threshold Exit Short"

        # 2) Cash 상태라면 신규 진입을 판단합니다.
        if current_position == "Cash":
            if cooldown_remaining > 0:
                # 익절/손절 직후에는 최소 4캔들 동안 신규 진입을 금지해 과매매를 줄입니다.
                cooldown_blocked = 1
                action_reason = "Cooldown"
            elif y_pred > ENTRY_LONG_THRESHOLD:
                current_position = "Long"
                entry_price = close_price
                fee_events += 1
                position_change_events += 1
                long_entry = 1
                action_reason = "Enter Long"
            elif y_pred < ENTRY_SHORT_THRESHOLD:
                current_position = "Short"
                entry_price = close_price
                fee_events += 1
                position_change_events += 1
                short_entry = 1
                action_reason = "Enter Short"

        # 3) 다음 15분 캔들의 고가/저가로 익절/손절 도달 여부를 검사하고 실제 수익률을 계산합니다.
        gross_strategy_return = 0.0
        risk_exit_happened = False

        if current_position == "Long":
            if entry_price is None:
                raise ValueError("Long 포지션인데 entry_price가 없습니다.")

            take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)

            # 한 캔들 안에서 익절선과 손절선이 모두 닿으면 실제 선후를 알 수 없습니다.
            # 보수적인 백테스트를 위해 손절이 먼저 체결된 것으로 가정합니다.
            if next_low <= stop_loss_price:
                gross_strategy_return = -STOP_LOSS_PCT
                current_position = "Cash"
                entry_price = None
                fee_events += 1
                position_change_events += 1
                stop_loss_exit = 1
                risk_exit_happened = True
                action_reason = "Stop Loss Long"
            elif next_high >= take_profit_price:
                gross_strategy_return = TAKE_PROFIT_PCT
                current_position = "Cash"
                entry_price = None
                fee_events += 1
                position_change_events += 1
                take_profit_exit = 1
                risk_exit_happened = True
                action_reason = "Take Profit Long"
            else:
                gross_strategy_return = next_close / close_price - 1

        elif current_position == "Short":
            if entry_price is None:
                raise ValueError("Short 포지션인데 entry_price가 없습니다.")

            take_profit_price = entry_price * (1 - TAKE_PROFIT_PCT)
            stop_loss_price = entry_price * (1 + STOP_LOSS_PCT)

            # Short은 가격이 내려가면 수익, 올라가면 손실입니다.
            # 마찬가지로 한 캔들에서 양쪽이 모두 닿으면 손절 우선으로 보수적으로 처리합니다.
            if next_high >= stop_loss_price:
                gross_strategy_return = -STOP_LOSS_PCT
                current_position = "Cash"
                entry_price = None
                fee_events += 1
                position_change_events += 1
                stop_loss_exit = 1
                risk_exit_happened = True
                action_reason = "Stop Loss Short"
            elif next_low <= take_profit_price:
                gross_strategy_return = TAKE_PROFIT_PCT
                current_position = "Cash"
                entry_price = None
                fee_events += 1
                position_change_events += 1
                take_profit_exit = 1
                risk_exit_happened = True
                action_reason = "Take Profit Short"
            else:
                gross_strategy_return = -(next_close / close_price - 1)

        # 4) 수수료는 실제 주문이 발생한 횟수에만 차감합니다.
        # 진입 1회, 청산 1회, Long<->Short 직접 전환은 청산+진입으로 2회 수수료를 냅니다.
        fee_cost = fee_events * FEE_RATE
        net_strategy_return = gross_strategy_return - fee_cost

        # 5) 익절/손절 청산 직후에는 다음 4캔들 동안 신규 진입을 막습니다.
        if risk_exit_happened:
            cooldown_remaining = COOLDOWN_CANDLES
        elif cooldown_remaining > 0 and current_position == "Cash":
            cooldown_remaining -= 1

        records.append(
            {
                "starting_position": starting_position,
                "position_state": current_position,
                "position": position_to_number(current_position),
                "entry_price": entry_price if entry_price is not None else np.nan,
                "ai_gross_return": gross_strategy_return,
                "fee_events": fee_events,
                "fee_cost": fee_cost,
                "ai_strategy_return": net_strategy_return,
                "position_change": position_change_events,
                "long_entry": long_entry,
                "short_entry": short_entry,
                "take_profit_exit": take_profit_exit,
                "stop_loss_exit": stop_loss_exit,
                "threshold_exit": threshold_exit,
                "cooldown_blocked": cooldown_blocked,
                "cooldown_remaining": cooldown_remaining,
                "action_reason": action_reason,
            }
        )

    return pd.DataFrame(records, index=backtest_df.index)


def run_backtest(test_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Buy & Hold와 AI 선물 양방향 전략의 누적 수익률 곡선을 계산합니다."""
    backtest_df = pd.merge(test_df, market_df, on="timestamp", how="left")
    before_rows = len(backtest_df)
    backtest_df = backtest_df.dropna(
        subset=["close", "next_high", "next_low", "next_close", "btc_15m_return", "y_pred"]
    ).reset_index(drop=True)
    dropped_rows = before_rows - len(backtest_df)

    if backtest_df.empty:
        raise ValueError("백테스트에 사용할 시장 데이터가 없습니다. timestamp 정렬을 확인해주세요.")

    print(f"[백테스트] 시장 데이터 매칭 완료: {len(backtest_df):,}개 사용, {dropped_rows:,}개 제외")

    # Buy & Hold는 테스트 시작 시점부터 BTC를 계속 보유한다고 가정합니다.
    backtest_df["buy_hold_return"] = backtest_df["btc_15m_return"]
    backtest_df["buy_hold_equity"] = (1 + backtest_df["buy_hold_return"]).cumprod()

    legacy_trade_count = calculate_zero_threshold_trade_count(backtest_df["y_pred"])

    # AI 선물 전략은 캔들별 루프 안에서 포지션, 리스크 관리, 수수료를 모두 계산합니다.
    position_df = build_futures_positions(backtest_df)
    backtest_df = pd.concat([backtest_df, position_df], axis=1)

    backtest_df["ai_gross_equity"] = (1 + backtest_df["ai_gross_return"]).cumprod()
    backtest_df["ai_strategy_equity"] = (1 + backtest_df["ai_strategy_return"]).cumprod()

    trade_count = int(backtest_df["position_change"].sum())
    long_entry_count = int(backtest_df["long_entry"].sum())
    short_entry_count = int(backtest_df["short_entry"].sum())
    fee_event_count = int(backtest_df["fee_events"].sum())
    take_profit_count = int(backtest_df["take_profit_exit"].sum())
    stop_loss_count = int(backtest_df["stop_loss_exit"].sum())
    threshold_exit_count = int(backtest_df["threshold_exit"].sum())
    cooldown_block_count = int(backtest_df["cooldown_blocked"].sum())

    print(
        "[백테스트] 리스크 관리 설정: "
        f"롱 진입 {ENTRY_LONG_THRESHOLD * 100:.2f}%, "
        f"숏 진입 {ENTRY_SHORT_THRESHOLD * 100:.2f}%, "
        f"롱 청산 {EXIT_LONG_THRESHOLD * 100:.2f}%, "
        f"숏 청산 {EXIT_SHORT_THRESHOLD * 100:.2f}%"
    )
    print(
        "[백테스트] 익절/손절 설정: "
        f"익절 {TAKE_PROFIT_PCT * 100:.2f}%, "
        f"손절 {STOP_LOSS_PCT * 100:.2f}%, "
        f"쿨다운 {COOLDOWN_CANDLES}캔들"
    )
    print(f"[백테스트] 참고용 기존 0 기준 Long/Cash 로직 포지션 변경 횟수: {legacy_trade_count:,}회")
    print(f"[백테스트] AI 선물 양방향 전략 포지션 변경 횟수: {trade_count:,}회")
    print(f"[백테스트] 롱 진입 횟수: {long_entry_count:,}회")
    print(f"[백테스트] 숏 진입 횟수: {short_entry_count:,}회")
    print(f"[백테스트] 익절(Take Profit) 청산 횟수: {take_profit_count:,}회")
    print(f"[백테스트] 손절(Stop Loss) 청산 횟수: {stop_loss_count:,}회")
    print(f"[백테스트] 모델 예측 변화(Threshold) 일반 청산 횟수: {threshold_exit_count:,}회")
    print(f"[백테스트] 쿨다운으로 신규 진입이 막힌 캔들 수: {cooldown_block_count:,}개")
    print(f"[백테스트] 수수료 부과 주문 횟수: {fee_event_count:,}회")
    print(f"[백테스트] 적용 수수료: 선물 시장가 주문 1회당 {FEE_RATE * 100:.2f}%")
    return backtest_df


def print_performance(backtest_df: pd.DataFrame) -> None:
    """Buy & Hold와 AI 선물 양방향 전략의 최종 누적 수익률과 MDD를 출력합니다."""
    buy_hold_final = float(backtest_df["buy_hold_equity"].iloc[-1])
    ai_gross_final = float(backtest_df["ai_gross_equity"].iloc[-1])
    ai_final = float(backtest_df["ai_strategy_equity"].iloc[-1])

    buy_hold_cumulative_return = buy_hold_final - 1
    ai_gross_cumulative_return = ai_gross_final - 1
    ai_cumulative_return = ai_final - 1

    buy_hold_mdd = calculate_mdd(backtest_df["buy_hold_equity"])
    ai_mdd = calculate_mdd(backtest_df["ai_strategy_equity"])

    print("\n[성과 요약]")
    print(f"Buy & Hold 최종 누적 수익률: {buy_hold_cumulative_return * 100:.2f}% (자산배율 {buy_hold_final:.4f})")
    print(f"AI 선물 양방향 전략 수수료 차감 전 누적 수익률: {ai_gross_cumulative_return * 100:.2f}% (자산배율 {ai_gross_final:.4f})")
    print(f"AI 선물 양방향 전략 최종 누적 수익률: {ai_cumulative_return * 100:.2f}% (자산배율 {ai_final:.4f})")
    print(f"Buy & Hold MDD: {buy_hold_mdd * 100:.2f}%")
    print(f"AI 선물 양방향 전략 MDD: {ai_mdd * 100:.2f}%")


def plot_equity_curve(backtest_df: pd.DataFrame) -> None:
    """Buy & Hold와 AI 선물 양방향 전략의 누적 수익률 곡선을 그리고 이미지로 저장합니다."""
    BACKTEST_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Windows 환경에서 한글 제목/라벨이 깨지지 않도록 한글 폰트를 지정합니다.
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(13, 7))
    plt.plot(
        backtest_df["timestamp"],
        backtest_df["buy_hold_equity"],
        color="blue",
        linewidth=1.5,
        label="Buy & Hold",
    )
    plt.plot(
        backtest_df["timestamp"],
        backtest_df["ai_strategy_equity"],
        color="red",
        linewidth=1.5,
        label="AI Futures Long & Short",
    )
    plt.title("Backtest Equity Curve: Buy & Hold vs AI Futures Long & Short")
    plt.xlabel("Timestamp")
    plt.ylabel("Cumulative Return (1.0 = Initial Capital)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(BACKTEST_IMAGE_PATH, dpi=150)
    print(f"[저장] 백테스트 수익률 곡선을 저장했습니다: {BACKTEST_IMAGE_PATH}")
    plt.show()


def main() -> None:
    """데이터 로드, 모델 예측, 백테스트, 성과 출력, 그래프 저장을 순서대로 실행합니다."""
    feature_df = load_feature_data()
    market_df = load_raw_15m_market_data()
    test_df = select_test_set(feature_df)

    model = load_model()
    test_with_prediction = predict_test_returns(model, test_df)

    backtest_df = run_backtest(test_with_prediction, market_df)
    print_performance(backtest_df)
    plot_equity_curve(backtest_df)


if __name__ == "__main__":
    main()
