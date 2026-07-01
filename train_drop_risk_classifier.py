"""
Train an XGBoost classifier for next-hour BTC drop risk.

This model predicts whether the next 4 x 15-minute candles, i.e. the next hour,
will drop by the configured risk-target threshold. It is a Risk Guard model, not
a direct buy/sell signal model.

Input:
    data/processed/BTCUSDT_15m_risk_targets.csv

Outputs:
    model/xgb_drop_risk_classifier.json
    model/drop_risk_feature_importance.png
    data/processed/BTCUSDT_15m_drop_risk_predictions.csv

Run:
    py -3 train_drop_risk_classifier.py
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
MODEL_PATH = MODEL_DIR / "xgb_drop_risk_classifier.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "drop_risk_feature_importance.png"
PREDICTION_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_drop_risk_predictions.csv"

TARGET_COLUMN = "target_drop_next_4"
RANDOM_STATE = 42
THRESHOLDS = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

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

PREDICTION_COLUMNS = [
    "timestamp",
    "target_drop_next_4",
    "target_return_next_4",
    "target_volatility_high_next_4",
    "target_big_move_next_4",
    "pred_proba_no_drop",
    "pred_proba_drop",
    "pred_label",
]
OPTIONAL_PREDICTION_COLUMNS = ["target_return_next", "target_up_next"]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input file was not found: {DATA_PATH}")

    print("[purpose] This model predicts next-hour drop risk for risk management.")
    print("[purpose] It is not a direct buy/sell signal model.")
    print(f"[load] input file: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    required_columns = {"timestamp", TARGET_COLUMN}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Input file is missing required columns: {missing_columns}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        details = ", ".join(f"{column}={count}" for column, count in missing_counts.items())
        raise ValueError(f"Input data contains missing or invalid values: {details}")

    unique_targets = sorted(df[TARGET_COLUMN].unique().tolist())
    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0/1 values. Found: {unique_targets}")

    print(f"[load] rows={len(df):,}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [column for column in df.columns if column not in EXCLUDED_FEATURE_COLUMNS]
    if not feature_columns:
        raise ValueError("No feature columns were found after excluding timestamp, close, and labels.")

    print("[excluded columns]")
    for column in sorted(EXCLUDED_FEATURE_COLUMNS):
        print(f"  - {column}")

    print("[feature columns]")
    print(f"total={len(feature_columns):,}")
    for column in feature_columns:
        print(f"  - {column}")
    return feature_columns


def print_class_ratio(name: str, y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    ratios = y.value_counts(normalize=True).sort_index()
    no_drop_count = int(counts.get(0, 0))
    drop_count = int(counts.get(1, 0))
    no_drop_ratio = float(ratios.get(0, 0.0))
    drop_ratio = float(ratios.get(1, 0.0))
    print(
        f"[class] {name}: no_drop=0 {no_drop_count:,} ({no_drop_ratio:.2%}), "
        f"drop=1 {drop_count:,} ({drop_ratio:.2%})"
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
        raise ValueError("Train split has no positive drop samples.")

    scale_pos_weight = negative_count / positive_count
    print("[class weight]")
    print(f"negative_count={negative_count:,}")
    print(f"positive_count={positive_count:,}")
    print(f"scale_pos_weight={scale_pos_weight:.6f}")
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
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
    )

    print("[train] starting XGBClassifier training")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )
    print("[train] complete")
    return model


def evaluate_test(model: XGBClassifier, test_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    y_true = test_df[TARGET_COLUMN].astype(int)
    probabilities = model.predict_proba(test_df[feature_columns])
    pred_drop = probabilities[:, 1]
    pred_label = (pred_drop >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, pred_label),
        "precision": precision_score(y_true, pred_label, zero_division=0),
        "recall": recall_score(y_true, pred_label, zero_division=0),
        "f1": f1_score(y_true, pred_label, zero_division=0),
        "roc_auc": roc_auc_score(y_true, pred_drop),
        "average_precision": average_precision_score(y_true, pred_drop),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
    }

    print("\n[test metrics]")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.6f}")

    matrix = confusion_matrix(y_true, pred_label, labels=[0, 1])
    print("[confusion matrix] labels=[0 no_drop, 1 drop]")
    print(matrix)

    print("\n[pred_proba_drop distribution]")
    print(pd.Series(pred_drop, name="pred_proba_drop").describe().to_string())

    print_threshold_comparison(y_true, pred_drop)
    return probabilities


def print_threshold_comparison(y_true: pd.Series, pred_drop: np.ndarray) -> None:
    rows: list[dict[str, float | int]] = []
    actual_drop = y_true.to_numpy() == 1
    for threshold in THRESHOLDS:
        predicted_drop = pred_drop >= threshold
        actual_drop_capture_count = int((predicted_drop & actual_drop).sum())
        false_alarm_count = int((predicted_drop & ~actual_drop).sum())
        rows.append(
            {
                "threshold": threshold,
                "predicted_drop_count": int(predicted_drop.sum()),
                "predicted_drop_ratio": float(predicted_drop.mean()),
                "precision": precision_score(y_true, predicted_drop.astype(int), zero_division=0),
                "recall": recall_score(y_true, predicted_drop.astype(int), zero_division=0),
                "f1": f1_score(y_true, predicted_drop.astype(int), zero_division=0),
                "actual_drop_capture_count": actual_drop_capture_count,
                "false_alarm_count": false_alarm_count,
            }
        )

    threshold_df = pd.DataFrame(rows)
    print("\n[threshold comparison]")
    with pd.option_context("display.width", 180, "display.max_columns", None, "display.float_format", "{:.6f}".format):
        print(threshold_df.to_string(index=False))
    print("[note] Low thresholds usually increase recall and false alarms.")
    print("[note] High thresholds usually increase precision but can miss many drops.")
    print("[note] Risk Guard usage should balance recall and false alarms, not precision alone.")


def save_predictions(test_df: pd.DataFrame, probabilities: np.ndarray) -> None:
    pred_label = (probabilities[:, 1] >= 0.5).astype(int)
    output_columns = [
        column
        for column in [
            "timestamp",
            "target_drop_next_4",
            "target_return_next_4",
            "target_volatility_high_next_4",
            "target_big_move_next_4",
            *OPTIONAL_PREDICTION_COLUMNS,
        ]
        if column in test_df.columns
    ]
    prediction_df = test_df[output_columns].copy()
    prediction_df["pred_proba_no_drop"] = probabilities[:, 0]
    prediction_df["pred_proba_drop"] = probabilities[:, 1]
    prediction_df["pred_label"] = pred_label

    ordered_columns = [column for column in PREDICTION_COLUMNS if column in prediction_df.columns]
    ordered_columns += [column for column in OPTIONAL_PREDICTION_COLUMNS if column in prediction_df.columns]
    prediction_df = prediction_df[ordered_columns]

    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(PREDICTION_PATH, index=False)
    print(f"[save] predictions rows={len(prediction_df):,} path={PREDICTION_PATH}")


def plot_feature_importance(model: XGBClassifier, feature_columns: list[str]) -> pd.DataFrame:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\n[feature importance]")
    print(importance_df.to_string(index=False))

    plot_df = importance_df.sort_values("importance", ascending=True)
    plt.figure(figsize=(11, 7))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.title("XGBoost Next-Hour Drop Risk Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
    plt.close()
    print(f"[save] feature importance plot: {FEATURE_IMPORTANCE_PATH}")
    return importance_df


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
    importance_df = plot_feature_importance(model, feature_columns)
    save_model(model)
    save_predictions(test_df, probabilities)
    print("\n[top 10 feature importance]")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, ImportError) as error:
        print(f"[error] {error}")
        sys.exit(1)
