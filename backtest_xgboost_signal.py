"""
Backtest long/cash strategies from XGBoost prediction probabilities.

Input:
    data/processed/BTCUSDT_15m_predictions.csv

Outputs:
    data/processed/xgboost_threshold_backtest_summary.csv
    data/processed/BTCUSDT_15m_backtest.csv
    model/xgboost_signal_equity_curve.png

Run:
    python backtest_xgboost_signal.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MPL_CONFIG_DIR = SCRIPT_DIR / ".matplotlib_cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

try:
    import matplotlib.pyplot as plt
except ImportError as error:
    raise ImportError(
        "matplotlib is not installed. Please run `pip install -r requirements.txt` first."
    ) from error


BASE_DIR = SCRIPT_DIR
PREDICTION_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_predictions.csv"
SUMMARY_PATH = BASE_DIR / "data" / "processed" / "xgboost_threshold_backtest_summary.csv"
DETAIL_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_backtest.csv"
EQUITY_CURVE_PATH = BASE_DIR / "model" / "xgboost_signal_equity_curve.png"

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]
FEE_PER_TRADE = 0.0006
REQUIRED_COLUMNS = [
    "timestamp",
    "target_up_next",
    "target_return_next",
    "pred_proba_down",
    "pred_proba_up",
    "pred_label",
]


def load_predictions() -> pd.DataFrame:
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file was not found: {PREDICTION_PATH}\n"
            "Please run `python train_xgboost_multi_timeframe.py` first."
        )

    print(f"[load] reading predictions: {PREDICTION_PATH}")
    df = pd.read_csv(PREDICTION_PATH)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Prediction file is missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for column in ["target_up_next", "target_return_next", "pred_proba_down", "pred_proba_up", "pred_label"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        details = ", ".join(f"{column}={count}" for column, count in missing_counts.items())
        raise ValueError(f"Prediction data contains missing or invalid values: {details}")

    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    print(f"[load] rows={len(df):,}")
    print(f"[load] period={df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")
    return df


def max_drawdown(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1
    return float(drawdown.min())


def calculate_trade_returns(df: pd.DataFrame) -> list[float]:
    trade_returns: list[float] = []
    in_trade = False
    compounded_return = 1.0

    for _, row in df.iterrows():
        position = int(row["position"])
        strategy_return = float(row["gross_strategy_return"])

        if position == 1:
            in_trade = True
            compounded_return *= 1 + strategy_return
        elif in_trade:
            trade_returns.append(compounded_return - 1)
            in_trade = False
            compounded_return = 1.0

    if in_trade:
        trade_returns.append(compounded_return - 1)

    return trade_returns


def backtest_threshold(predictions: pd.DataFrame, threshold: float) -> tuple[dict[str, float], pd.DataFrame]:
    df = predictions.copy()
    signal_column = f"signal_{threshold:.2f}"
    df[signal_column] = (df["pred_proba_up"] >= threshold).astype(int)

    # The model signal is shifted one bar so the decision is applied only to the
    # next available return, avoiding lookahead bias.
    df["position"] = df[signal_column].shift(1).fillna(0).astype(int)
    df["gross_strategy_return"] = df["position"] * df["target_return_next"]
    df["trade"] = df["position"].diff().abs().fillna(df["position"]).astype(int)
    df["fee"] = df["trade"] * FEE_PER_TRADE
    df["strategy_return"] = df["gross_strategy_return"] - df["fee"]
    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()
    df["buy_hold_equity_curve"] = (1 + df["target_return_next"]).cumprod()

    trade_returns = calculate_trade_returns(df)
    winning_trades = [value for value in trade_returns if value > 0]
    losing_trades = [value for value in trade_returns if value < 0]

    number_of_trades = int(df["trade"].sum())
    completed_trades = len(trade_returns)
    win_rate = len(winning_trades) / completed_trades if completed_trades else 0.0
    average_win = float(np.mean(winning_trades)) if winning_trades else 0.0
    average_loss = float(np.mean(losing_trades)) if losing_trades else 0.0
    expected_value = float(np.mean(trade_returns)) if trade_returns else 0.0
    gross_profit = float(np.sum(winning_trades)) if winning_trades else 0.0
    gross_loss = abs(float(np.sum(losing_trades))) if losing_trades else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf if gross_profit > 0 else 0.0

    summary = {
        "threshold": threshold,
        "total_return": float(df["equity_curve"].iloc[-1] - 1),
        "buy_and_hold_return": float(df["buy_hold_equity_curve"].iloc[-1] - 1),
        "max_drawdown": max_drawdown(df["equity_curve"]),
        "number_of_trades": number_of_trades,
        "completed_trades": completed_trades,
        "win_rate": win_rate,
        "average_win": average_win,
        "average_loss": average_loss,
        "expected_value_per_trade": expected_value,
        "profit_factor": profit_factor,
    }
    return summary, df


def print_summary(summary_df: pd.DataFrame) -> None:
    print("\n[summary by threshold]")
    display_columns = [
        "threshold",
        "total_return",
        "buy_and_hold_return",
        "max_drawdown",
        "number_of_trades",
        "win_rate",
        "average_win",
        "average_loss",
        "expected_value_per_trade",
        "profit_factor",
    ]
    with pd.option_context("display.width", 180, "display.max_columns", None, "display.float_format", "{:.6f}".format):
        print(summary_df[display_columns].to_string(index=False))


def select_best_threshold(summary_df: pd.DataFrame) -> float:
    sortable = summary_df.copy()
    sortable["profit_factor_sort"] = sortable["profit_factor"].replace(np.inf, 1_000_000.0)
    best_row = sortable.sort_values(
        ["total_return", "max_drawdown", "profit_factor_sort"],
        ascending=[False, False, False],
    ).iloc[0]
    best_threshold = float(best_row["threshold"])
    print(f"[best] threshold={best_threshold:.2f} selected by highest total return")
    return best_threshold


def save_summary(summary_df: pd.DataFrame) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"[save] summary: {SUMMARY_PATH}")


def save_detail(detail_df: pd.DataFrame, threshold: float) -> None:
    output = detail_df.copy()
    output.insert(1, "threshold", threshold)
    columns = [
        "timestamp",
        "threshold",
        "target_return_next",
        "pred_proba_up",
        "position",
        "trade",
        "fee",
        "gross_strategy_return",
        "strategy_return",
        "equity_curve",
        "buy_hold_equity_curve",
    ]
    output[columns].to_csv(DETAIL_PATH, index=False)
    print(f"[save] best-threshold detail rows={len(output):,} path={DETAIL_PATH}")


def plot_equity_curve(detail_df: pd.DataFrame, threshold: float) -> None:
    EQUITY_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    plt.plot(detail_df["timestamp"], detail_df["equity_curve"], label=f"XGBoost threshold {threshold:.2f}")
    plt.plot(detail_df["timestamp"], detail_df["buy_hold_equity_curve"], label="Buy & Hold", alpha=0.8)
    plt.title("XGBoost Signal Equity Curve")
    plt.xlabel("Timestamp")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(EQUITY_CURVE_PATH, dpi=150)
    plt.close()
    print(f"[save] equity curve: {EQUITY_CURVE_PATH}")


def main() -> None:
    predictions = load_predictions()

    summaries: list[dict[str, float]] = []
    details_by_threshold: dict[float, pd.DataFrame] = {}
    for threshold in THRESHOLDS:
        summary, detail_df = backtest_threshold(predictions, threshold)
        summaries.append(summary)
        details_by_threshold[threshold] = detail_df

    summary_df = pd.DataFrame(summaries)
    print_summary(summary_df)
    save_summary(summary_df)

    best_threshold = select_best_threshold(summary_df)
    best_detail = details_by_threshold[best_threshold]
    save_detail(best_detail, best_threshold)
    plot_equity_curve(best_detail, best_threshold)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"[error] {error}")
        sys.exit(1)
