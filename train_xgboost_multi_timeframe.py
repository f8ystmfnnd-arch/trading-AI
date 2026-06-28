"""
Train an XGBoost classifier with multi-timeframe BTCUSDT features.

Input:
    data/processed/BTCUSDT_15m_features.csv

Outputs:
    model/xgb_multi_timeframe_classifier.json
    model/multi_timeframe_feature_importance.png
    data/processed/BTCUSDT_15m_predictions.csv

Run:
    python train_xgboost_multi_timeframe.py
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
DATA_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_features.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "xgb_multi_timeframe_classifier.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "multi_timeframe_feature_importance.png"
PREDICTION_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_predictions.csv"

TARGET_COLUMN = "target_up_next"
RETURN_ANALYSIS_COLUMN = "target_return_next"
EXCLUDED_FEATURE_COLUMNS = {"timestamp", TARGET_COLUMN, RETURN_ANALYSIS_COLUMN}
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature file was not found: {DATA_PATH}\n"
            "Please run `python create_features_multi_timeframe.py` first."
        )

    print(f"[load] reading feature data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    required_columns = {"timestamp", TARGET_COLUMN, RETURN_ANALYSIS_COLUMN}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Feature file is missing required columns: {missing_columns}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        details = ", ".join(f"{column}={count}" for column, count in missing_counts.items())
        raise ValueError(
            "Feature data contains missing or invalid values. "
            f"Please regenerate/check the processed feature file. Missing counts: {details}"
        )

    unique_targets = sorted(df[TARGET_COLUMN].unique().tolist())
    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0/1 values. Found: {unique_targets}")

    print(f"[load] rows={len(df):,}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [column for column in df.columns if column not in EXCLUDED_FEATURE_COLUMNS]
    if not feature_columns:
        raise ValueError("No feature columns were found after excluding timestamp and targets.")

    print("[features] feature columns:")
    for column in feature_columns:
        print(f"  - {column}")
    print(f"[features] total={len(feature_columns):,}")
    return feature_columns


def print_class_ratio(name: str, y: pd.Series) -> None:
    ratios = y.value_counts(normalize=True).sort_index()
    counts = y.value_counts().sort_index()
    down_count = int(counts.get(0, 0))
    up_count = int(counts.get(1, 0))
    down_ratio = float(ratios.get(0, 0.0))
    up_ratio = float(ratios.get(1, 0.0))
    print(
        f"[class] {name}: down=0 {down_count:,} ({down_ratio:.2%}), "
        f"up=1 {up_count:,} ({up_ratio:.2%})"
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
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
    )

    print("[train] fitting XGBClassifier without shuffling")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )
    print("[train] complete")
    return model


def evaluate_split(model: XGBClassifier, name: str, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    X = df[feature_columns]
    y_true = df[TARGET_COLUMN].astype(int)
    probabilities = model.predict_proba(X)
    pred_up = probabilities[:, 1]
    pred_label = (pred_up >= 0.5).astype(int)

    try:
        roc_auc = roc_auc_score(y_true, pred_up)
    except ValueError:
        roc_auc = np.nan

    metrics = {
        "accuracy": accuracy_score(y_true, pred_label),
        "precision": precision_score(y_true, pred_label, zero_division=0),
        "recall": recall_score(y_true, pred_label, zero_division=0),
        "f1": f1_score(y_true, pred_label, zero_division=0),
        "roc_auc": roc_auc,
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
    }

    print(f"\n[metrics] {name}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.6f}")

    return probabilities


def save_predictions(model: XGBClassifier, test_df: pd.DataFrame, feature_columns: list[str]) -> None:
    probabilities = model.predict_proba(test_df[feature_columns])
    pred_label = (probabilities[:, 1] >= 0.5).astype(int)

    prediction_df = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"],
            "target_up_next": test_df[TARGET_COLUMN].astype(int),
            "target_return_next": test_df[RETURN_ANALYSIS_COLUMN],
            "pred_proba_down": probabilities[:, 0],
            "pred_proba_up": probabilities[:, 1],
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
    plt.title("XGBoost Multi-Timeframe Feature Importance")
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
    evaluate_split(model, "validation", validation_df, feature_columns)
    evaluate_split(model, "test", test_df, feature_columns)

    plot_feature_importance(model, feature_columns)
    save_model(model)
    save_predictions(model, test_df, feature_columns)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"[error] {error}")
        sys.exit(1)
