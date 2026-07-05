"""Train a high-volatility risk classifier with technical indicator features.

This is a Risk Guard AI experiment, not a direct buy/sell signal model.
Technical indicators are used as market-regime features, not trading formulas.

Inputs:
    data/processed/BTCUSDT_15m_features_with_indicators.csv
    data/processed/BTCUSDT_15m_risk_targets.csv

Outputs:
    model/xgb_volatility_classifier_with_indicators.json
    model/volatility_feature_importance_with_indicators.csv
    data/processed/BTCUSDT_15m_volatility_predictions_with_indicators.csv

Run:
    py -3 train_volatility_classifier_with_indicators.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    raise ImportError("scikit-learn is not installed. Please run `pip install -r requirements.txt`.") from error

try:
    from xgboost import XGBClassifier
except ImportError as error:
    raise ImportError("xgboost is not installed. Please run `pip install -r requirements.txt`.") from error


BASE_DIR = Path(__file__).resolve().parent
FEATURE_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_features_with_indicators.csv"
RISK_TARGET_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_risk_targets.csv"

MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "xgb_volatility_classifier_with_indicators.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "volatility_feature_importance_with_indicators.csv"
PREDICTION_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_volatility_predictions_with_indicators.csv"

TARGET_COLUMN = "target_volatility_high_next_4"
RANDOM_STATE = 42
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

INDICATOR_FEATURE_COLUMNS = [
    "bb_middle_20",
    "bb_upper_20",
    "bb_lower_20",
    "bb_width_20",
    "bb_percent_b_20",
    "bb_breakout_up_20",
    "bb_breakout_down_20",
    "atr_14",
    "atr_ratio_14",
    "ma_20_slope",
    "ma_60_slope",
    "ma_120_slope",
    "rsi_14_slope",
]

BASELINE_RESULTS = {
    "roc_auc": 0.788064,
    "average_precision": 0.599165,
    "accuracy": 0.739374,
    "precision": 0.518282,
    "recall": 0.638955,
    "log_loss": 0.525593,
}


def require_file(path: Path, label: str) -> None:
    if path.exists():
        return
    raise FileNotFoundError(f"Missing {label} file: {path}")


def load_csv(path: Path, label: str) -> pd.DataFrame:
    require_file(path, label)
    print(f"[load] {label}: {path}")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{label} is missing required column: timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    return df.reset_index(drop=True)


def target_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("target_")]


def load_dataset() -> pd.DataFrame:
    features = load_csv(FEATURE_PATH, "features with indicators")
    risk_targets = load_csv(RISK_TARGET_PATH, "risk targets")

    print(f"[rows] input features rows: {len(features):,}")
    print(f"[rows] input risk target rows: {len(risk_targets):,}")

    if TARGET_COLUMN not in risk_targets.columns:
        raise ValueError(f"Risk target file is missing required target column: {TARGET_COLUMN}")

    risk_columns = ["timestamp"]
    if "close" in risk_targets.columns and "close" not in risk_columns:
        risk_columns.append("close")
    risk_columns.extend(
        column
        for column in target_columns(risk_targets)
        if column not in risk_columns and column not in features.columns
    )
    if TARGET_COLUMN not in risk_columns:
        risk_columns.append(TARGET_COLUMN)

    merged = features.merge(risk_targets[risk_columns], on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged = merged.replace([np.inf, -np.inf], np.nan)

    print(f"[rows] rows after merge: {len(merged):,}")

    unique_targets = sorted(merged[TARGET_COLUMN].dropna().unique().tolist())
    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0/1 values. Found: {unique_targets}")

    return merged


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_columns = {
        "timestamp",
        "close",
        *target_columns(df),
    }
    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns and pd.api.types.is_numeric_dtype(df[column])
    ]

    if TARGET_COLUMN in feature_columns or any(column.startswith("target_") for column in feature_columns):
        raise ValueError("Target or future-return columns leaked into feature columns.")
    if not feature_columns:
        raise ValueError("No numeric feature columns were found after exclusions.")

    print("[excluded] non-feature / target columns:")
    for column in sorted(excluded_columns):
        if column in df.columns:
            print(f"  - {column}")

    present_indicators = [column for column in INDICATOR_FEATURE_COLUMNS if column in feature_columns]
    missing_indicators = [column for column in INDICATOR_FEATURE_COLUMNS if column not in feature_columns]

    print("[features] indicator features included:")
    for column in present_indicators:
        print(f"  - {column}")
    if missing_indicators:
        print("[features] indicator features missing:")
        for column in missing_indicators:
            print(f"  - {column}")

    print(f"[features] total feature count: {len(feature_columns):,}")
    return feature_columns


def drop_missing_rows(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    before_count = len(df)
    cleaned = df.dropna(subset=[*feature_columns, TARGET_COLUMN]).copy()
    removed_count = before_count - len(cleaned)
    print(f"[clean] rows removed for NaN before training: {removed_count:,}")
    return cleaned.reset_index(drop=True)


def print_class_ratio(name: str, y: pd.Series) -> None:
    ratio = float(y.mean()) if len(y) else 0.0
    positives = int(y.sum())
    print(f"[class] {name}: rows={len(y):,}, positive={positives:,}, positive_ratio={ratio:.6f}")


def split_time_series(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_count = len(df)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    if train_end <= 0 or validation_end <= train_end or validation_end >= row_count:
        raise ValueError("Not enough rows to split into 70/15/15 train/validation/test sets.")

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    print(f"[split] train rows: {len(train_df):,}")
    print(f"[split] validation rows: {len(validation_df):,}")
    print(f"[split] test rows: {len(test_df):,}")
    print_class_ratio("train", train_df[TARGET_COLUMN].astype(int))
    print_class_ratio("validation", validation_df[TARGET_COLUMN].astype(int))
    print_class_ratio("test", test_df[TARGET_COLUMN].astype(int))
    return train_df, validation_df, test_df


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    if positive_count == 0:
        raise ValueError("Train split has no positive high-volatility samples.")
    scale_pos_weight = negative_count / positive_count
    print(f"[class weight] scale_pos_weight: {scale_pos_weight:.6f}")
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
        scale_pos_weight=calculate_scale_pos_weight(y_train),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
    )

    print("[train] fitting XGBClassifier with time-series split and no shuffle")
    model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=False)
    print("[train] complete")
    return model


def evaluate_test(model: XGBClassifier, test_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    y_true = test_df[TARGET_COLUMN].astype(int)
    probabilities = model.predict_proba(test_df[feature_columns])
    pred_high = probabilities[:, 1]
    pred_label = (pred_high >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, pred_label),
        "precision": precision_score(y_true, pred_label, zero_division=0),
        "recall": recall_score(y_true, pred_label, zero_division=0),
        "f1": f1_score(y_true, pred_label, zero_division=0),
        "roc_auc": roc_auc_score(y_true, pred_high),
        "average_precision": average_precision_score(y_true, pred_high),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
    }

    print("\n[metrics] test")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.6f}")

    print("\n[metrics] confusion matrix labels=[0 low_vol, 1 high_vol]")
    print(confusion_matrix(y_true, pred_label, labels=[0, 1]))

    print("\n[analysis] threshold precision/recall summary")
    print(" threshold  predicted_high_count  precision   recall")
    for threshold in THRESHOLDS:
        threshold_label = (pred_high >= threshold).astype(int)
        precision = precision_score(y_true, threshold_label, zero_division=0)
        recall = recall_score(y_true, threshold_label, zero_division=0)
        predicted_count = int(threshold_label.sum())
        print(f" {threshold:8.2f} {predicted_count:21,} {precision:10.6f} {recall:8.6f}")

    print("\n[comparison] previous 5y high-vol baseline vs with-indicators test")
    for metric_name, baseline_value in BASELINE_RESULTS.items():
        current_value = metrics[metric_name]
        delta = current_value - baseline_value
        print(
            f"  {metric_name}: baseline={baseline_value:.6f}, "
            f"with_indicators={current_value:.6f}, delta={delta:+.6f}"
        )

    return probabilities


def save_predictions(test_df: pd.DataFrame, probabilities: np.ndarray) -> None:
    prediction_df = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"],
            TARGET_COLUMN: test_df[TARGET_COLUMN].astype(int),
            "pred_proba_low_vol": probabilities[:, 0],
            "pred_proba_high_vol": probabilities[:, 1],
            "pred_label": (probabilities[:, 1] >= 0.5).astype(int),
        }
    )

    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(PREDICTION_PATH, index=False)
    print(f"[save] predictions: {PREDICTION_PATH}")


def save_feature_importance(model: XGBClassifier, feature_columns: list[str]) -> None:
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    print("\n[feature importance] top 20")
    print(importance_df.head(20).to_string(index=False))
    print(f"[save] feature importance: {FEATURE_IMPORTANCE_PATH}")


def save_model(model: XGBClassifier) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"[save] model: {MODEL_PATH}")


def main() -> None:
    print("[purpose] BTC Market Regime & Risk Guard AI")
    print("[purpose] Predict next-hour high-volatility risk, not buy/sell direction.")

    df = load_dataset()
    feature_columns = get_feature_columns(df)
    df = drop_missing_rows(df, feature_columns)

    train_df, validation_df, test_df = split_time_series(df)
    model = train_model(train_df, validation_df, feature_columns)
    probabilities = evaluate_test(model, test_df, feature_columns)

    save_model(model)
    save_feature_importance(model, feature_columns)
    save_predictions(test_df, probabilities)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"[error] {error}")
        sys.exit(1)
