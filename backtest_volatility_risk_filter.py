"""
Backtest high-volatility risk-off filters using volatility classifier probabilities.

This is not a directional trading signal backtest. It checks whether reducing
exposure during predicted high-volatility regimes can protect equity and reduce
maximum drawdown versus buy-and-hold.

Run:
    py -3 backtest_volatility_risk_filter.py
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
PREDICTION_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_volatility_predictions.csv"
RISK_TARGET_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_risk_targets.csv"
RESAMPLED_15M_PATH = BASE_DIR / "data" / "resampled" / "BTCUSDT_15m.csv"
SUMMARY_PATH = BASE_DIR / "data" / "processed" / "volatility_risk_filter_backtest_summary.csv"
DETAIL_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_volatility_risk_filter_backtest.csv"
EQUITY_CURVE_PATH = BASE_DIR / "model" / "volatility_risk_filter_equity_curve.png"

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
REPRESENTATIVE_THRESHOLDS = [0.50, 0.60, 0.70]
FEE_RATE = 0.0006
REDUCED_EXPOSURE = 0.5
BARS_PER_YEAR = 365 * 24 * 4
BARS_PER_DAY = 24 * 4
BARS_PER_7D = 7 * BARS_PER_DAY

RISK_COLUMNS = [
    "timestamp",
    "target_return_next",
    "target_return_next_4",
    "target_volatility_high_next_4",
    "target_drop_next_4",
    "target_big_move_next_4",
]


def load_predictions() -> pd.DataFrame:
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(f"Prediction file was not found: {PREDICTION_PATH}")

    df = pd.read_csv(PREDICTION_PATH)
    if "timestamp" not in df.columns:
        raise ValueError("Prediction file must contain timestamp column.")

    proba_column = find_high_vol_probability_column(df)
    df = df[["timestamp", proba_column]].copy()
    df = df.rename(columns={proba_column: "pred_proba_high_vol"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["pred_proba_high_vol"] = pd.to_numeric(df["pred_proba_high_vol"], errors="coerce")
    df = df.dropna(subset=["timestamp", "pred_proba_high_vol"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df


def find_high_vol_probability_column(df: pd.DataFrame) -> str:
    if "pred_proba_high_vol" in df.columns:
        return "pred_proba_high_vol"

    candidates = [
        column
        for column in df.columns
        if "proba" in column.lower() and any(token in column.lower() for token in ["high", "vol", "1"])
    ]
    if len(candidates) == 1:
        print(f"[load] using inferred high-vol probability column: {candidates[0]}")
        return candidates[0]

    raise ValueError(
        "Could not find high-volatility class=1 probability column. "
        "Expected pred_proba_high_vol or a clear proba/high/vol column. "
        f"Columns: {df.columns.tolist()}"
    )


def load_risk_targets() -> pd.DataFrame:
    if not RISK_TARGET_PATH.exists():
        raise FileNotFoundError(f"Risk target file was not found: {RISK_TARGET_PATH}")

    df = pd.read_csv(RISK_TARGET_PATH)
    if "timestamp" not in df.columns:
        raise ValueError("Risk target file must contain timestamp column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    missing_columns = [column for column in RISK_COLUMNS if column not in df.columns]
    if missing_columns:
        print(f"[load] risk target file is missing columns: {missing_columns}")
        df = add_missing_returns_from_close(df, missing_columns)

    missing_columns = [column for column in RISK_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Required risk columns are still missing after fallback: {missing_columns}")

    output = df[RISK_COLUMNS].copy()
    for column in RISK_COLUMNS:
        if column != "timestamp":
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def add_missing_returns_from_close(df: pd.DataFrame, missing_columns: list[str]) -> pd.DataFrame:
    needs_return_next = "target_return_next" in missing_columns
    if not needs_return_next:
        return df

    if "close" not in df.columns:
        if not RESAMPLED_15M_PATH.exists():
            raise FileNotFoundError(
                f"Need close to compute target_return_next, but {RESAMPLED_15M_PATH} was not found."
            )
        close_df = pd.read_csv(RESAMPLED_15M_PATH, usecols=["timestamp", "close"])
        close_df["timestamp"] = pd.to_datetime(close_df["timestamp"], utc=True, errors="coerce")
        close_df["close"] = pd.to_numeric(close_df["close"], errors="coerce")
        close_df = close_df.dropna(subset=["timestamp", "close"])
        close_df = close_df.sort_values("timestamp").drop_duplicates(subset="timestamp")
        df = df.merge(close_df, on="timestamp", how="left", validate="one_to_one")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["target_return_next"] = df["close"].shift(-1) / df["close"] - 1
    return df


def build_dataset() -> pd.DataFrame:
    predictions = load_predictions()
    risk_targets = load_risk_targets()

    print(f"[input] predictions: {PREDICTION_PATH}")
    print(f"[input] risk targets: {RISK_TARGET_PATH}")
    print(f"[input] predictions rows: {len(predictions):,}")

    df = predictions.merge(risk_targets, on="timestamp", how="inner", validate="one_to_one")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("Merged backtest dataset is empty.")

    print(f"[merge] merged rows: {len(df):,}")
    print(f"[period] start={df['timestamp'].iloc[0]} end={df['timestamp'].iloc[-1]}")
    print("[analysis] pred_proba_high_vol distribution")
    print(df["pred_proba_high_vol"].describe().to_string())
    return df


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def rolling_compounded_return(returns: pd.Series, window: int) -> float:
    rolled = (1 + returns).rolling(window=window, min_periods=1).apply(np.prod, raw=True) - 1
    return float(rolled.min())


def annualized_return(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    total_return = float(equity.iloc[-1] - 1)
    if total_return <= -1:
        return -1.0
    years = len(equity) / BARS_PER_YEAR
    return float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * np.sqrt(BARS_PER_YEAR))


def evaluate_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    prefix: str,
    threshold: float,
    risk_position: float,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    detail = pd.DataFrame(index=df.index)
    raw_position = np.where(df["pred_proba_high_vol"] >= threshold, risk_position, 1.0)
    position = pd.Series(raw_position, index=df.index).shift(1).fillna(1.0)
    position_change = position.diff().abs().fillna(0.0)
    fee = position_change * FEE_RATE
    strategy_return = position * df["target_return_next"] - fee
    equity = equity_curve(strategy_return)
    dd = drawdown(equity)

    risk_off = position < 1.0
    actual_high_vol = df["target_volatility_high_next_4"] == 1
    actual_drop = df["target_drop_next_4"] == 1
    actual_big_move = df["target_big_move_next_4"] == 1

    total_high_vol = int(actual_high_vol.sum())
    total_drop = int(actual_drop.sum())
    total_big_move = int(actual_big_move.sum())
    total_risk_off = int(risk_off.sum())

    avoided_high_vol_ratio = safe_ratio((risk_off & actual_high_vol).sum(), total_high_vol)
    missed_high_vol_ratio = safe_ratio((~risk_off & actual_high_vol).sum(), total_high_vol)
    risk_off_precision = safe_ratio((risk_off & actual_high_vol).sum(), total_risk_off)
    risk_off_recall = avoided_high_vol_ratio
    avoided_drop_ratio = safe_ratio((risk_off & actual_drop).sum(), total_drop)
    risk_off_big_move_ratio = safe_ratio((risk_off & actual_big_move).sum(), total_big_move)

    ann_vol = annualized_volatility(strategy_return)
    ann_ret = annualized_return(equity)
    summary = {
        "strategy_name": strategy_name,
        "threshold": threshold,
        "total_return": float(equity.iloc[-1] - 1),
        "buy_and_hold_return": float(df["buy_and_hold_equity"].iloc[-1] - 1),
        "excess_return_vs_bh": float((equity.iloc[-1] - 1) - (df["buy_and_hold_equity"].iloc[-1] - 1)),
        "max_drawdown": float(dd.min()),
        "buy_and_hold_max_drawdown": float(df["buy_and_hold_drawdown"].min()),
        "mdd_improvement": float(dd.min() - df["buy_and_hold_drawdown"].min()),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_like": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "exposure_ratio": float(position.mean()),
        "risk_off_ratio": float(risk_off.mean()),
        "number_of_position_changes": int((position_change > 0).sum()),
        "total_fee_paid": float(fee.sum()),
        "average_position": float(position.mean()),
        "worst_1d_return": rolling_compounded_return(strategy_return, BARS_PER_DAY),
        "worst_7d_return": rolling_compounded_return(strategy_return, BARS_PER_7D),
        "avoided_high_vol_ratio": float(avoided_high_vol_ratio),
        "missed_high_vol_ratio": float(missed_high_vol_ratio),
        "risk_off_precision": float(risk_off_precision),
        "risk_off_recall": float(risk_off_recall),
        "avoided_drop_ratio": float(avoided_drop_ratio),
        "risk_off_during_big_move_ratio": float(risk_off_big_move_ratio),
    }

    detail[f"{prefix}_position"] = position
    detail[f"{prefix}_return"] = strategy_return
    detail[f"{prefix}_equity"] = equity
    detail[f"{prefix}_drawdown"] = dd
    return summary, detail


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def threshold_label(threshold: float) -> str:
    return f"t{int(round(threshold * 100)):03d}"


def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = df[
        [
            "timestamp",
            "target_return_next",
            "target_return_next_4",
            "target_volatility_high_next_4",
            "target_drop_next_4",
            "target_big_move_next_4",
            "pred_proba_high_vol",
        ]
    ].copy()
    detail["buy_and_hold_return"] = detail["target_return_next"]
    detail["buy_and_hold_equity"] = equity_curve(detail["buy_and_hold_return"])
    detail["buy_and_hold_drawdown"] = drawdown(detail["buy_and_hold_equity"])

    summaries: list[dict[str, float | str]] = []
    for threshold in THRESHOLDS:
        label = threshold_label(threshold)
        cash_summary, cash_detail = evaluate_strategy(
            detail,
            strategy_name="cash_filter",
            prefix=f"cash_{label}",
            threshold=threshold,
            risk_position=0.0,
        )
        half_summary, half_detail = evaluate_strategy(
            detail,
            strategy_name="half_exposure_filter",
            prefix=f"half_{label}",
            threshold=threshold,
            risk_position=REDUCED_EXPOSURE,
        )
        summaries.extend([cash_summary, half_summary])
        detail = pd.concat([detail, cash_detail, half_detail], axis=1)

    return pd.DataFrame(summaries), detail


def print_console_report(df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    print("\n[buy & hold]")
    print(f"total_return: {df['buy_and_hold_equity'].iloc[-1] - 1:.6f}")
    print(f"max_drawdown: {df['buy_and_hold_drawdown'].min():.6f}")

    display_columns = [
        "strategy_name",
        "threshold",
        "total_return",
        "buy_and_hold_return",
        "excess_return_vs_bh",
        "max_drawdown",
        "mdd_improvement",
        "exposure_ratio",
        "risk_off_ratio",
        "number_of_position_changes",
        "total_fee_paid",
        "avoided_high_vol_ratio",
        "risk_off_precision",
        "risk_off_recall",
        "avoided_drop_ratio",
    ]
    print("\n[summary table]")
    with pd.option_context("display.width", 240, "display.max_columns", None, "display.float_format", "{:.6f}".format):
        print(summary_df[display_columns].to_string(index=False))

    highest_total_return = summary_df.sort_values("total_return", ascending=False).iloc[0]
    best_mdd_improvement = summary_df.sort_values("mdd_improvement", ascending=False).iloc[0]
    print("\n[best candidates]")
    print("highest_total_return:")
    print(highest_total_return.to_string())
    print("\nbest_mdd_improvement:")
    print(best_mdd_improvement.to_string())
    print("\n[note] Interpret high total return carefully if exposure_ratio is very low.")
    print("[note] This backtest is for risk filtering, not direct buy/sell signal generation.")


def save_outputs(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    detail_df.to_csv(DETAIL_PATH, index=False)
    print(f"[save] summary CSV: {SUMMARY_PATH}")
    print(f"[save] detail CSV: {DETAIL_PATH}")


def plot_equity_curves(detail_df: pd.DataFrame) -> None:
    EQUITY_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for axis, kind, prefix in [(axes[0], "cash_filter", "cash"), (axes[1], "half_exposure_filter", "half")]:
        axis.plot(detail_df["timestamp"], detail_df["buy_and_hold_equity"], label="Buy & Hold", linewidth=1.5)
        for threshold in REPRESENTATIVE_THRESHOLDS:
            label = threshold_label(threshold)
            axis.plot(
                detail_df["timestamp"],
                detail_df[f"{prefix}_{label}_equity"],
                label=f"{kind} {threshold:.2f}",
                linewidth=1.0,
            )
        axis.set_title(f"Volatility Risk Filter Equity Curve - {kind}")
        axis.set_ylabel("Equity")
        axis.grid(True, alpha=0.25)
        axis.legend()

    axes[-1].set_xlabel("Timestamp")
    plt.tight_layout()
    plt.savefig(EQUITY_CURVE_PATH, dpi=150)
    plt.close()
    print(f"[save] equity curve PNG: {EQUITY_CURVE_PATH}")


def main() -> None:
    merged_df = build_dataset()
    summary_df, detail_df = run_backtest(merged_df)
    print_console_report(detail_df, summary_df)
    save_outputs(summary_df, detail_df)
    plot_equity_curves(detail_df)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, ImportError) as error:
        print(f"[error] {error}")
        sys.exit(1)
