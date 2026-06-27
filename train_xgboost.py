"""
BTC/USDT 15분봉 피처 데이터로 XGBoost 회귀 모델을 학습하고 평가합니다.

입력:
    data/btc_15m_5y_features.csv

출력:
    model/xgb_model.json

실행:
    python train_xgboost.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Matplotlib은 최초 import 시 폰트 캐시를 만드는데, 권한 문제가 생기지 않도록
# 프로젝트 내부의 쓰기 가능한 캐시 폴더를 사용하게 설정합니다.
SCRIPT_DIR = Path(__file__).resolve().parent
MPL_CONFIG_DIR = SCRIPT_DIR / ".matplotlib_cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

try:
    import matplotlib.pyplot as plt
except ImportError as error:
    raise ImportError("matplotlib이 설치되어 있지 않습니다. 먼저 `pip install -r requirements.txt`를 실행해주세요.") from error

try:
    import sklearn  # noqa: F401
except ImportError as error:
    raise ImportError("XGBRegressor 실행에 필요한 scikit-learn이 없습니다. `pip install -r requirements.txt`를 실행해주세요.") from error

try:
    from xgboost import XGBRegressor
except ImportError as error:
    raise ImportError("xgboost가 설치되어 있지 않습니다. 먼저 `pip install -r requirements.txt`를 실행해주세요.") from error


# =========================
# 파일 경로 설정
# =========================
BASE_DIR = SCRIPT_DIR
DATA_PATH = BASE_DIR / "data" / "btc_15m_5y_features.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "xgb_model.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.png"


# =========================
# 학습 설정
# =========================
TRAIN_RATIO = 0.8
RANDOM_STATE = 42

# timestamp, target, 원본 OHLCV 컬럼은 학습 피처에서 제외합니다.
# 특히 close 같은 절대 가격은 시계열 레벨 변화 때문에 모델을 왜곡할 수 있습니다.
EXCLUDED_FEATURE_COLUMNS = {"timestamp", "target", "open", "high", "low", "close", "volume"}


def load_dataset() -> pd.DataFrame:
    """피처 CSV를 불러오고 시간순 정렬 및 결측치 점검을 수행합니다."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"학습 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")

    print(f"[로드] 학습용 피처 데이터를 불러옵니다: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    if "timestamp" not in df.columns:
        raise ValueError("timestamp 컬럼이 없습니다. 시계열 순서 보존을 위해 timestamp가 필요합니다.")
    if "target" not in df.columns:
        raise ValueError("target 컬럼이 없습니다. 회귀 모델이 맞춰야 할 정답 컬럼이 필요합니다.")

    # timestamp를 datetime으로 변환한 뒤 정렬하여 시간 순서를 확실하게 유지합니다.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 무한대 값은 모델 학습에 들어갈 수 없으므로 NaN으로 바꾸고 제거합니다.
    before_rows = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    dropped_rows = before_rows - len(df)

    print(f"[로드 완료] 전체 행 수: {len(df):,}개, 결측/무한대 제거 행 수: {dropped_rows:,}개")
    return df


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """학습 피처 X와 정답 y를 분리합니다."""
    feature_columns = [column for column in df.columns if column not in EXCLUDED_FEATURE_COLUMNS]

    if not feature_columns:
        raise ValueError("학습에 사용할 피처 컬럼이 없습니다. 입력 CSV의 컬럼을 확인해주세요.")

    # X는 모델이 보고 학습할 기술적 지표들이고, y는 4시간 뒤 수익률 target입니다.
    X = df[feature_columns].copy()
    y = df["target"].copy()

    print(f"[피처 분리] 사용 피처 {len(feature_columns)}개: {feature_columns}")
    print("[피처 분리] timestamp와 절대 가격/거래량 원본 컬럼은 피처에서 제외했습니다.")
    return X, y, feature_columns


def time_series_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """무작위 섞기 없이 앞 80%를 Train, 뒤 20%를 Test로 나눕니다."""
    split_index = int(len(X) * TRAIN_RATIO)

    if split_index <= 0 or split_index >= len(X):
        raise ValueError("Train/Test 분할을 수행하기에 데이터가 충분하지 않습니다.")

    # 시계열에서는 미래 데이터가 과거 학습에 섞이면 Data Leakage가 생기므로 절대 shuffle하지 않습니다.
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"[분할] Train: {len(X_train):,}개 ({TRAIN_RATIO:.0%}), Test: {len(X_test):,}개 ({1 - TRAIN_RATIO:.0%})")
    print(f"[분할] Train 기간: {timestamps.iloc[0]} ~ {timestamps.iloc[split_index - 1]}")
    print(f"[분할] Test  기간: {timestamps.iloc[split_index]} ~ {timestamps.iloc[-1]}")
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """과적합을 줄이도록 보수적인 하이퍼파라미터로 XGBoost 회귀 모델을 학습합니다."""
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
    )

    print("[학습] XGBoost 회귀 모델 학습을 시작합니다.")
    model.fit(X_train, y_train)
    print("[학습 완료] 모델 학습이 끝났습니다.")
    return model


def evaluate_model(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """Test 데이터에 대한 예측값을 만들고 회귀 성능과 방향성 승률을 출력합니다."""
    print("[평가] Test 데이터로 미래 수익률을 예측합니다.")
    y_pred = model.predict(X_test)

    errors = y_test.to_numpy() - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    # R2는 평균만 예측하는 모델 대비 얼마나 더 잘 설명하는지를 나타냅니다.
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_test.to_numpy() - np.mean(y_test.to_numpy())) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    # 방향성 승률은 실제 수익률과 예측 수익률의 부호가 같은 비율입니다.
    actual_direction = np.where(y_test.to_numpy() >= 0, 1, -1)
    predicted_direction = np.where(y_pred >= 0, 1, -1)
    direction_accuracy = float(np.mean(actual_direction == predicted_direction) * 100)

    print("\n[모델 성능]")
    print(f"MAE  (평균절대오차): {mae:.8f}")
    print(f"RMSE (평균제곱근오차): {rmse:.8f}")
    print(f"R2 Score: {r2:.6f}")
    print(f"방향성 예측 승률: {direction_accuracy:.2f}%")

    return y_pred


def plot_feature_importance(model: XGBRegressor, feature_columns: list[str]) -> None:
    """모델이 중요하게 사용한 피처를 막대그래프로 시각화합니다."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=True)

    print("\n[피처 중요도]")
    print(importance_df.sort_values("importance", ascending=False).to_string(index=False))

    # Windows 환경에서 한글 제목이 깨지지 않도록 기본 한글 폰트를 지정합니다.
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title("XGBoost 피처 중요도")
    plt.xlabel("중요도")
    plt.ylabel("피처")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
    print(f"[저장] 피처 중요도 그래프를 이미지로 저장했습니다: {FEATURE_IMPORTANCE_PATH}")
    plt.show()


def save_model(model: XGBRegressor) -> None:
    """학습된 XGBoost 모델을 실시간 예측 대시보드에서 재사용할 수 있도록 JSON으로 저장합니다."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"[저장] 학습된 모델을 저장했습니다: {MODEL_PATH}")


def main() -> None:
    """데이터 로드, 시계열 분할, 모델 학습, 평가, 중요도 시각화, 모델 저장을 순서대로 실행합니다."""
    df = load_dataset()
    X, y, feature_columns = split_features_and_target(df)
    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y, df["timestamp"])

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, feature_columns)
    save_model(model)


if __name__ == "__main__":
    main()
