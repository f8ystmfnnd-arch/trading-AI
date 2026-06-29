"""
Train an XGBoost classifier for next-hour high-volatility risk.

This is a risk-management model, not a direct buy/sell signal model. Its output
is intended for position sizing, entry blocking, strategy selection, and Tilt
Guard style filters.

Input:
    data/processed/BTCUSDT_15m_risk_targets.csv

Outputs:
    model/xgb_volatility_classifier.json
    model/volatility_feature_importance.png
    data/processed/BTCUSDT_15m_volatility_predictions.csv

Run:
    python train_volatility_classifier.py
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

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as error:
    raise ImportError(
        "scikit-learn is not installed. Please run `pip install -r requirements.txt` first."
    ) from error

try:
    from xgboost import XGBClassifier
except ImportError as error:
    raise ImportError(
        "xgboost is not installed. Please run `pip install -r requirements.txt` first."
    ) from error


BASE_DIR = SCRIPT_DIR
DATA_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_risk_targets.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "xgb_volatility_classifier.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "volatility_feature_importance.png"
PREDICTION_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_volatility_predictions.csv"

TARGET_COLUMN = "target_volatility_high_next_4"
RANDOM_STATE = 42
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

EXCLUDED_FEATURE_COLUMNS = {
    "timestamp",
    "close",
    "target_return_next",
    "target_up_next",
    "target_return_next_4",
    "target_abs_return_next_4",
    "target_volatility_next_4",
    "target_big_move_next_4",
    "target_drop_next_4",
    "target_pump_next_4",
    "target_volatility_high_next_4",
}

PIPELINE_STEPS = [
    "python collect_bybit_1m.py",
    "python resample_ohlcv.py",
    "python create_features_multi_timeframe.py",
    "python create_risk_targets.py",
    "python train_volatility_classifier.py",
]


def pipeline_hint() -> str:
    return "Please run the data pipeline in this order:\n  " + "\n  ".join(PIPELINE_STEPS)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Risk target file was not found: {DATA_PATH}\n{pipeline_hint()}")

    print("[purpose] This model predicts next-hour high-volatility risk for risk management.")
    print("[purpose] It is not a direct buy/sell signal model.")
    print(f"[load] reading risk target data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    required_columns = {"timestamp", TARGET_COLUMN}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Risk target file is missing required columns: {missing_columns}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        details = ", ".join(f"{column}={count}" for column, count in missing_counts.items())
        raise ValueError(
            "Risk target data contains missing or invalid values. "
            f"Please regenerate/check the file. Missing counts: {details}"
        )

    unique_targets = sorted(df[TARGET_COLUMN].unique().tolist())
    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0/1 values. Found: {unique_targets}")

    print(f"[load] rows={len(df):,}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [column for column in df.columns if column not in EXCLUDED_FEATURE_COLUMNS]
    if not feature_columns:
        raise ValueError("No feature columns were found after excluding timestamp, close, and labels.")

    print("[excluded] target/label/non-feature columns:")
    for column in sorted(EXCLUDED_FEATURE_COLUMNS):
        print(f"  - {column}")

    print("[features] feature columns used by the volatility risk model:")
    for column in feature_columns:
        print(f"  - {column}")
    print(f"[features] total={len(feature_columns):,}")
    return feature_columns


def print_class_ratio(name: str, y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    ratios = y.value_counts(normalize=True).sort_index()
    low_count = int(counts.get(0, 0))
    high_count = int(counts.get(1, 0))
    low_ratio = float(ratios.get(0, 0.0))
    high_ratio = float(ratios.get(1, 0.0))
    print(
        f"[class] {name}: low_vol=0 {low_count:,} ({low_ratio:.2%}), "
        f"high_vol=1 {high_count:,} ({high_ratio:.2%})"
    )


def print_period(name: str, df: pd.DataFrame) -> None:
    print(
        f"[split] {name}: rows={len(df):,}, "
        f"start={df['timestamp'].iloc[0]}, end={df['timestamp'].iloc[-1]}"
    )
    print_class_ratio(name, df[TARGET_COLUMN])


def split_time_series(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_count = len(df)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)

    if train_end <= 0 or validation_end <= train_end or validation_end >= row_count:
        raise ValueError("Not enough rows to split into 70% train, 15% validation, 15% test.")

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    print_period("train", train_df)
    print_period("validation", validation_df)
    print_period("test", test_df)
    return train_df, validation_df, test_df


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    if positive_count == 0:
        raise ValueError("Train split has no positive high-volatility samples.")

    scale_pos_weight = negative_count / positive_count
    print(
        f"[class weight] negative_count={negative_count:,}, "
        f"positive_count={positive_count:,}, scale_pos_weight={scale_pos_weight:.6f}"
    )
    return scale_pos_weight


def train_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: list[str],
) -> XGBClassifier:
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_validation = validation_df[feature_columns]
    y_validation = validation_df[TARGET_COLUMN].astype(int)
    scale_pos_weight = calculate_scale_pos_weight(y_train)

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=350,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
    )

    print("[train] fitting XGBClassifier without shuffling")
    print("[train] eval_set is validation data; output is used to monitor risk-model behavior.")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )
    print("[train] complete")
    return model


def predict_probabilities(model: XGBClassifier, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return model.predict_proba(df[feature_columns])


def evaluate_test(model: XGBClassifier, test_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    y_true = test_df[TARGET_COLUMN].astype(int)
    probabilities = predict_probabilities(model, test_df, feature_columns)
    pred_high = probabilities[:, 1]
    pred_label = (pred_high >= 0.5).astype(int)

    try:
        roc_auc = roc_auc_score(y_true, pred_high)
    except ValueError:
        roc_auc = np.nan

    try:
        average_precision = average_precision_score(y_true, pred_high)
    except ValueError:
        average_precision = np.nan

    metrics = {
        "accuracy": accuracy_score(y_true, pred_label),
        "precision": precision_score(y_true, pred_label, zero_division=0),
        "recall": recall_score(y_true, pred_label, zero_division=0),
        "f1": f1_score(y_true, pred_label, zero_division=0),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
    }

    print("\n[metrics] test")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.6f}")

    matrix = confusion_matrix(y_true, pred_label, labels=[0, 1])
    print("[metrics] confusion matrix labels=[0 low_vol, 1 high_vol]")
    print(matrix)

    print("\n[analysis] pred_proba_high_vol distribution")
    print(pd.Series(pred_high, name="pred_proba_high_vol").describe().to_string())

    print("\n[analysis] threshold comparison for risk guard usage")
    print(" threshold  predicted_high_count  precision   recall       f1")
    for threshold in THRESHOLDS:
        threshold_label = (pred_high >= threshold).astype(int)
        precision = precision_score(y_true, threshold_label, zero_division=0)
        recall = recall_score(y_true, threshold_label, zero_division=0)
        f1 = f1_score(y_true, threshold_label, zero_division=0)
        predicted_count = int(threshold_label.sum())
        print(f" {threshold:8.2f} {predicted_count:21,} {precision:10.6f} {recall:8.6f} {f1:8.6f}")

    return probabilities


def save_predictions(test_df: pd.DataFrame, probabilities: np.ndarray) -> None:
    pred_label = (probabilities[:, 1] >= 0.5).astype(int)
    prediction_df = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"],
            "target_volatility_high_next_4": test_df[TARGET_COLUMN].astype(int),
            "pred_proba_low_vol": probabilities[:, 0],
            "pred_proba_high_vol": probabilities[:, 1],
            "pred_label": pred_label,
        }
    )

    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(PREDICTION_PATH, index=False)
    print(f"[save] predictions rows={len(prediction_df):,} path={PREDICTION_PATH}")


def plot_feature_importance(model: XGBClassifier, feature_columns: list[str]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=True)

    print("\n[feature importance]")
    print(importance_df.sort_values("importance", ascending=False).to_string(index=False))

    plt.figure(figsize=(11, 7))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title("XGBoost High-Volatility Risk Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
    plt.close()
    print(f"[save] feature importance plot: {FEATURE_IMPORTANCE_PATH}")


def save_model(model: XGBClassifier) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"[save] model: {MODEL_PATH}")


def main() -> None:
    df = load_dataset()
    feature_columns = get_feature_columns(df)
    train_df, validation_df, test_df = split_time_series(df)

    model = train_model(train_df, validation_df, feature_columns)
    probabilities = evaluate_test(model, test_df, feature_columns)

    plot_feature_importance(model, feature_columns)
    save_model(model)
    save_predictions(test_df, probabilities)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"[error] {error}")
        sys.exit(1)
