"""TradingView-style dashboard for BTC Market Regime & Risk Guard AI."""

from __future__ import annotations

import json
import urllib.request
from collections import deque
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


BASE_DIR = Path(__file__).resolve().parents[1]

CANDLE_15M_PATH = BASE_DIR / "data" / "resampled" / "BTCUSDT_15m.csv"
FEATURE_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_features_with_indicators.csv"
VOLATILITY_PREDICTION_PATH = (
    BASE_DIR / "data" / "processed" / "BTCUSDT_15m_volatility_predictions_with_indicators.csv"
)
DROP_RISK_PREDICTION_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_drop_risk_predictions.csv"
SIMILARITY_DIR = BASE_DIR / "data" / "processed" / "similarity"

BYBIT_REST_TICKER_URL = "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT"
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"
BYBIT_WS_TOPIC = "tickers.BTCUSDT"
REST_FALLBACK_MS = 5000

TIMEFRAME_PATHS = {
    "1m": BASE_DIR / "data" / "raw" / "BTCUSDT_1m.csv",
    "5m": BASE_DIR / "data" / "resampled" / "BTCUSDT_5m.csv",
    "15m": BASE_DIR / "data" / "resampled" / "BTCUSDT_15m.csv",
    "1h": BASE_DIR / "data" / "resampled" / "BTCUSDT_1h.csv",
    "4h": BASE_DIR / "data" / "resampled" / "BTCUSDT_4h.csv",
    "1d": BASE_DIR / "data" / "resampled" / "BTCUSDT_1d.csv",
}

OHLC_COLUMNS = ["timestamp", "open", "high", "low", "close"]
MA_INDICATOR_COLUMNS = ["ma_20", "ma_60", "ma_120"]
BB_INDICATOR_COLUMNS = ["bb_upper_20", "bb_lower_20"]
DEFAULT_INDICATOR_COLUMNS = [*MA_INDICATOR_COLUMNS, *BB_INDICATOR_COLUMNS]
DEFAULT_HIGH_VOL_MARKER_THRESHOLD = 0.80
DEFAULT_DROP_RISK_MARKER_THRESHOLD = 0.25
DEFAULT_NO_TRADE_HIGH_VOL_THRESHOLD = 0.85
DEFAULT_NO_TRADE_DROP_RISK_THRESHOLD = 0.30
RISK_OVERLAY_TIMEFRAME = "15m"
RISK_OVERLAY_CANDLE_COUNT = 1000
MICRO_VIEW_TIMEFRAME = "1m"
MICRO_VIEW_CANDLE_COUNT = 1000
DEFAULT_FAST_DROP_5M_THRESHOLD = -0.004
DEFAULT_FAST_PUMP_5M_THRESHOLD = 0.004
DEFAULT_VOLUME_SPIKE_RATIO_THRESHOLD = 3.0
DEFAULT_RANGE_SPIKE_RATIO_THRESHOLD = 3.0
DEFAULT_VOLATILITY_SPIKE_PERCENTILE = 0.90
TIMEFRAME_BUCKET_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

UI_LANG = "ko"
UI_TEXT = {
    "ko": {
        "view_mode": "화면 모드",
        "view_modes": {
            "Normal Dashboard": "기본 대시보드",
            "15m Risk Overlay": "15분봉 리스크 오버레이",
            "1m Live Micro View": "1분봉 미세흐름 보기",
        },
        "websocket_live": "실시간 가격 반영",
        "fallback_rest": "REST 보조 조회 간격: 5초",
        "data_status": "데이터 상태",
        "timeframe": "시간봉",
        "recent_candles": "최근 캔들 수",
        "overlay_controls": "오버레이 설정",
        "micro_overlay_controls": "미세흐름 오버레이 설정",
        "show_ma_lines": "이동평균선 표시",
        "show_bollinger_bands": "볼린저밴드 표시",
        "show_high_vol_markers": "고변동 마커 표시",
        "show_drop_risk_markers": "급락위험 마커 표시",
        "show_no_trade_markers": "진입금지 마커 표시",
        "show_risk_event_table": "리스크 이벤트 표 표시",
        "show_marker_events_only_in_table": "차트에 찍힌 이벤트만 표에 표시",
        "high_vol_marker_threshold": "고변동 마커 기준값",
        "drop_risk_marker_threshold": "급락위험 마커 기준값",
        "no_trade_high_vol_threshold": "진입금지 고변동 기준값",
        "no_trade_drop_risk_threshold": "진입금지 급락위험 기준값",
        "show_volatility_spike_markers": "변동성 급증 마커 표시",
        "show_volume_spike_markers": "거래량 급증 마커 표시",
        "show_fast_drop_markers": "단기 급락 마커 표시",
        "show_fast_pump_markers": "단기 급등 마커 표시",
        "show_micro_event_table": "미세흐름 이벤트 표 표시",
        "fast_drop_5m_threshold": "5분 급락 기준 수익률",
        "fast_pump_5m_threshold": "5분 급등 기준 수익률",
        "volume_spike_ratio_threshold": "거래량 급증 기준 배율",
        "range_spike_ratio_threshold": "캔들 변동폭 급증 기준 배율",
        "volatility_spike_percentile": "변동성 급증 분위 기준값",
        "high_vol_probability": "고변동 가능성",
        "drop_risk_probability": "급락 위험",
        "atr_ratio": "평균 변동폭",
        "bb_width": "볼린저밴드 폭",
        "action_hint": "시장 상태",
        "high_vol_help": "가까운 미래에 가격 변동이 평소보다 커질 것으로 모델이 판단한 확률입니다.",
        "drop_risk_help": "가까운 미래에 큰 가격 하락이 발생할 것으로 모델이 판단한 확률입니다.",
        "atr_ratio_help": "ATR을 현재 가격으로 나눈 값입니다. 사건 발생 확률이 아니라 현재 가격 대비 평균 변동폭입니다.",
        "bb_width_help": "상단 볼린저밴드와 하단 볼린저밴드 사이의 폭을 현재 가격 대비 비율로 나타낸 값입니다. 확률이나 위험 점수가 아닙니다.",
        "action_hint_help": "고변동 가능성, 급락 위험 등 여러 위험 지표를 종합한 현재 시장 상태입니다.",
        "live_btcusdt_price": "실시간 BTCUSDT 가격",
        "return_1m": "1분 수익률",
        "return_5m": "5분 수익률",
        "return_15m": "15분 수익률",
        "volume_ratio_60": "거래량 비율(60)",
        "volatility_20": "변동성(20)",
        "action_15m": "상위 리스크 상태(15분봉)",
        "normal_caption": "실시간 가격 반영은 화면 참고용입니다. Risk Guard 판단은 15분봉 피처와 모델을 기준으로 봅니다.",
        "risk_overlay_title": "BTCUSDT · 15분봉 리스크 오버레이",
        "risk_overlay_caption": (
            "최근 1,000개의 15분봉 BTCUSDT 캔들 위에 리스크 오버레이를 표시합니다. "
            "마커는 매수/매도 신호가 아니라 위험 참고용입니다. "
            "차트가 복잡해지지 않도록 위험 상태가 시작되는 지점만 표시합니다."
        ),
        "micro_view_title": "BTCUSDT · 1분봉 미세흐름 보기",
        "micro_view_caption": (
            "최근 1,000개의 1분봉 BTCUSDT 캔들에서 급격한 변동, 거래량 급증, 단기 급락/급등을 보여줍니다. "
            "가격 예측 모델이나 매수/매도 신호가 아니라 리스크 참고용입니다."
        ),
        "recent_risk_events": "최근 리스크 이벤트",
        "no_risk_events": "현재 기준값에서 최근 1,000개 15분봉 안에 표시할 리스크 이벤트가 없습니다.",
        "recent_micro_events": "최근 미세흐름 이벤트",
        "no_micro_events": "현재 기준값에서 최근 1,000개 1분봉 안에 표시할 미세흐름 이벤트가 없습니다.",
        "chart_debug": "차트 디버그",
        "no_chart_debug": "표시할 차트 디버그 정보가 없습니다.",
        "baseline_title": "지표 추가 모델 기준 성능 비교",
        "baseline_caption": (
            "기술적 지표는 기계적인 매매 공식이 아니라 시장 국면을 판단하기 위한 피처로 사용합니다.\n\n"
            "- `roc_auc`: `0.788064` -> `0.795049`\n"
            "- `average_precision`: `0.599165` -> `0.610102`\n"
            "- `log_loss`: `0.525593` -> `0.512464`"
        ),
        "live_price_line": "실시간 BTCUSDT",
        "event_labels": {
            "NO_TRADE": "진입금지",
            "HIGH_VOL": "고변동",
            "DROP_RISK": "급락",
            "FAST_DROP": "급락",
            "FAST_PUMP": "급등",
            "VOLUME_SPIKE": "거래량",
            "VOL_SPIKE": "변동성",
            "RANGE_SPIKE": "변동폭",
        },
        "action_labels": {
            "NORMAL": "정상",
            "CAUTION": "주의",
            "NO_TRADE": "진입금지",
        },
        "action_descriptions": {
            "NORMAL": "감지된 시장 위험이 낮은 상태",
            "CAUTION": "일부 위험 지표가 상승한 상태",
            "NO_TRADE": "시장 변동 또는 하락 위험이 높아 신규 진입을 피하는 상태",
        },
        "risk_event_columns": {
            "timestamp": "시각",
            "event": "이벤트",
            "pred_proba_high_vol": "고변동 확률",
            "pred_proba_drop": "급락위험 확률",
            "atr_ratio_14": "ATR 비율",
            "bb_width_20": "볼린저밴드 폭",
            "reason": "사유",
        },
        "micro_event_columns": {
            "timestamp": "시각",
            "event": "이벤트",
            "close": "종가",
            "return_1m": "1분 수익률",
            "return_5m": "5분 수익률",
            "return_15m": "15분 수익률",
            "range_1m": "1분 변동폭",
            "volume_ratio_60": "거래량 비율(60)",
            "volatility_20": "변동성(20)",
            "reason": "사유",
        },
    },
    "en": {
        "view_mode": "View Mode",
        "view_modes": {
            "Normal Dashboard": "Normal Dashboard",
            "15m Risk Overlay": "15m Risk Overlay",
            "1m Live Micro View": "1m Live Micro View",
        },
        "websocket_live": "WebSocket live",
        "fallback_rest": "Fallback REST polling interval: 5s",
        "data_status": "Data status",
        "timeframe": "Timeframe",
        "recent_candles": "Recent candles",
        "overlay_controls": "Overlay Controls",
        "micro_overlay_controls": "Micro Overlay Controls",
        "show_ma_lines": "Show MA lines",
        "show_bollinger_bands": "Show Bollinger Bands",
        "show_high_vol_markers": "Show High-vol markers",
        "show_drop_risk_markers": "Show Drop-risk markers",
        "show_no_trade_markers": "Show NO_TRADE markers",
        "show_risk_event_table": "Show Risk event table",
        "show_marker_events_only_in_table": "Show marker events only in table",
        "high_vol_marker_threshold": "High-vol marker threshold",
        "drop_risk_marker_threshold": "Drop-risk marker threshold",
        "no_trade_high_vol_threshold": "NO_TRADE high-vol threshold",
        "no_trade_drop_risk_threshold": "NO_TRADE drop-risk threshold",
        "show_volatility_spike_markers": "Show volatility spike markers",
        "show_volume_spike_markers": "Show volume spike markers",
        "show_fast_drop_markers": "Show fast drop markers",
        "show_fast_pump_markers": "Show fast pump markers",
        "show_micro_event_table": "Show micro event table",
        "fast_drop_5m_threshold": "Fast drop 5m return threshold",
        "fast_pump_5m_threshold": "Fast pump 5m return threshold",
        "volume_spike_ratio_threshold": "Volume spike ratio threshold",
        "range_spike_ratio_threshold": "Range spike ratio threshold",
        "volatility_spike_percentile": "Volatility spike percentile threshold",
        "high_vol_probability": "High-vol probability",
        "drop_risk_probability": "Drop risk",
        "atr_ratio": "Average move width",
        "bb_width": "Bollinger Band width",
        "action_hint": "Market state",
        "high_vol_help": "The model-estimated probability that near-future price movement will be larger than usual.",
        "drop_risk_help": "The model-estimated probability of a large near-future price drop.",
        "atr_ratio_help": "ATR divided by current price. This is a volatility ratio, not an event probability.",
        "bb_width_help": "The distance between upper and lower Bollinger Bands as a ratio of current price. This is not a probability or risk score.",
        "action_hint_help": "The current market state summarized from multiple risk indicators.",
        "live_btcusdt_price": "Live BTCUSDT price",
        "return_1m": "1m return",
        "return_5m": "5m return",
        "return_15m": "15m return",
        "volume_ratio_60": "volume_ratio_60",
        "volatility_20": "volatility_20",
        "action_15m": "Higher timeframe risk state (15m)",
        "normal_caption": "Live updates are visual references. Risk Guard decisions are based on 15m features/models.",
        "risk_overlay_title": "BTCUSDT · 15m Risk Overlay",
        "risk_overlay_caption": (
            "This view shows the latest 1,000 15-minute BTCUSDT candles with Risk Guard overlays. "
            "Markers are risk references, not buy/sell signals. "
            "Markers are compressed to risk-state start points to avoid visual clutter."
        ),
        "micro_view_title": "BTCUSDT · 1m Live Micro View",
        "micro_view_caption": (
            "This view shows the latest 1,000 1-minute BTCUSDT candles with micro risk events. "
            "It is not a price prediction model or a buy/sell signal."
        ),
        "recent_risk_events": "Recent Risk Events",
        "no_risk_events": "No risk events found in the latest 1,000 15m candles for the current thresholds.",
        "recent_micro_events": "Recent Micro Events",
        "no_micro_events": "No micro events found in the latest 1,000 1m candles for the current thresholds.",
        "chart_debug": "Chart debug",
        "no_chart_debug": "No chart debug data available.",
        "baseline_title": "Indicator model baseline comparison",
        "baseline_caption": (
            "Technical indicators are used as market-regime features, not mechanical trading formulas.\n\n"
            "- `roc_auc`: `0.788064` -> `0.795049`\n"
            "- `average_precision`: `0.599165` -> `0.610102`\n"
            "- `log_loss`: `0.525593` -> `0.512464`"
        ),
        "live_price_line": "Live BTCUSDT",
        "event_labels": {
            "NO_TRADE": "NO TRADE",
            "HIGH_VOL": "HIGH VOL",
            "DROP_RISK": "DROP",
            "FAST_DROP": "DROP",
            "FAST_PUMP": "PUMP",
            "VOLUME_SPIKE": "VOLM",
            "VOL_SPIKE": "VOL",
            "RANGE_SPIKE": "RANGE",
        },
        "action_labels": {
            "NORMAL": "NORMAL",
            "CAUTION": "CAUTION",
            "NO_TRADE": "NO_TRADE",
        },
        "action_descriptions": {
            "NORMAL": "Low detected market risk",
            "CAUTION": "Some risk indicators are elevated",
            "NO_TRADE": "Volatility or downside risk is high enough to avoid new entries",
        },
        "risk_event_columns": {
            "timestamp": "timestamp",
            "event": "event",
            "pred_proba_high_vol": "pred_proba_high_vol",
            "pred_proba_drop": "pred_proba_drop",
            "atr_ratio_14": "atr_ratio_14",
            "bb_width_20": "bb_width_20",
            "reason": "reason",
        },
        "micro_event_columns": {
            "timestamp": "timestamp",
            "event": "event",
            "close": "close",
            "return_1m": "return_1m",
            "return_5m": "return_5m",
            "return_15m": "return_15m",
            "range_1m": "range_1m",
            "volume_ratio_60": "volume_ratio_60",
            "volatility_20": "volatility_20",
            "reason": "reason",
        },
    },
}


st.set_page_config(page_title="BTC Market Regime & Risk Guard AI", layout="wide")


def t(key: str) -> Any:
    lang_text = UI_TEXT.get(UI_LANG, UI_TEXT["ko"])
    return lang_text.get(key, UI_TEXT["ko"].get(key, key))


def display_action_hint(value: Any) -> str:
    return t("action_labels").get(str(value), str(value))


def display_action_with_description(value: Any) -> str:
    action = str(value)
    label = display_action_hint(action)
    description = t("action_descriptions").get(action)
    return f"{label} - {description}" if description else label


def display_event(value: Any) -> str:
    return t("event_labels").get(str(value), str(value))


def display_table(df: pd.DataFrame, columns: list[str], mapping_key: str) -> pd.DataFrame:
    table = df.loc[:, columns].copy()
    table["timestamp"] = table["timestamp"].astype(str)
    if "event" in table.columns:
        table["event"] = table["event"].map(display_event)
    return table.rename(columns=t(mapping_key))


@st.cache_data(show_spinner=False)
def read_csv_tail(path: str, limit: int) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as file:
        header = file.readline()
        rows = deque(file, maxlen=limit)
    if not header:
        return pd.DataFrame()
    return pd.read_csv(StringIO(header + "".join(rows)))


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    return df.reset_index(drop=True)


def numeric_clean(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.replace([float("inf"), float("-inf")], pd.NA)


def load_ohlcv(timeframe: str, limit: int) -> tuple[pd.DataFrame | None, str | None]:
    path = TIMEFRAME_PATHS[timeframe]
    if not path.exists():
        return None, f"Missing {timeframe} candle file: {path}"
    try:
        df = normalize_timestamp(read_csv_tail(str(path), limit))
    except Exception as exc:
        return None, f"Failed to read {timeframe} candle file: {exc}"

    missing = [column for column in OHLC_COLUMNS if column not in df.columns]
    if missing:
        return None, f"{timeframe} candle file is missing columns: {', '.join(missing)}"

    df = numeric_clean(df.loc[:, OHLC_COLUMNS], ["open", "high", "low", "close"])
    df = df.dropna(subset=OHLC_COLUMNS)
    if df.empty:
        return None, f"{timeframe} candle file has no usable OHLC rows."
    return df.reset_index(drop=True), None


def load_indicators_for_15m(
    limit: int,
    indicator_columns: list[str] | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    if not FEATURE_PATH.exists():
        return None, f"Missing 15m indicator file: {FEATURE_PATH}"
    try:
        df = normalize_timestamp(read_csv_tail(str(FEATURE_PATH), limit + 300))
    except Exception as exc:
        return None, f"Failed to read 15m indicator file: {exc}"

    selected = DEFAULT_INDICATOR_COLUMNS if indicator_columns is None else indicator_columns
    columns = ["timestamp", *[column for column in selected if column in df.columns]]
    if len(columns) <= 1:
        return None, "15m indicator file has no usable overlay columns."
    df = numeric_clean(df.loc[:, columns], [column for column in columns if column != "timestamp"])
    return df.reset_index(drop=True), None


def unix_seconds(series: pd.Series) -> pd.Series:
    return series.map(lambda value: int(pd.Timestamp(value).timestamp())).astype(int)


def candles_to_chart_data(candles: pd.DataFrame) -> list[dict[str, float | int]]:
    df = candles.loc[:, OHLC_COLUMNS].copy()
    df["time"] = unix_seconds(df["timestamp"])
    return [
        {
            "time": int(row.time),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        for row in df.itertuples(index=False)
    ]


def indicators_to_chart_data(
    indicators: pd.DataFrame | None,
    candles: pd.DataFrame,
    indicator_columns: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if indicators is None or indicators.empty:
        return {}
    merged = candles[["timestamp"]].merge(indicators, on="timestamp", how="left")
    merged["time"] = unix_seconds(merged["timestamp"])

    output: dict[str, list[dict[str, Any]]] = {}
    for column in indicator_columns:
        if column not in merged.columns:
            continue
        clean = merged.loc[merged[column].notna(), ["time", column]]
        output[column] = [
            {"time": int(row.time), "value": float(getattr(row, column))}
            for row in clean.itertuples(index=False)
        ]
    return output


def build_lightweight_chart_html(
    candles: pd.DataFrame,
    indicators: pd.DataFrame | None,
    timeframe: str,
    action_hint: str,
    live_enabled: bool,
    indicator_columns: list[str] | None = None,
    markers: list[dict[str, Any]] | None = None,
    current_price: float | None = None,
) -> str:
    candle_data = candles_to_chart_data(candles)
    selected_indicators = DEFAULT_INDICATOR_COLUMNS if indicator_columns is None else indicator_columns
    indicator_data = indicators_to_chart_data(indicators, candles, selected_indicators)
    latest_close = float(candles.iloc[-1]["close"]) if not candles.empty else None
    initial_live_price = current_price if current_price is not None else latest_close

    payload = {
        "symbol": "BTCUSDT",
        "timeframe": timeframe,
        "candles": candle_data,
        "indicators": indicator_data,
        "markers": markers or [],
        "livePrice": initial_live_price,
        "actionHint": display_action_hint(action_hint),
        "livePriceLineTitle": t("live_price_line"),
        "liveEnabled": bool(live_enabled),
        "bucketSeconds": TIMEFRAME_BUCKET_SECONDS[timeframe],
        "rangeKey": f"btc_chart_range_{timeframe}_{len(candle_data)}",
        "wsUrl": BYBIT_WS_URL,
        "wsTopic": BYBIT_WS_TOPIC,
        "restTickerUrl": BYBIT_REST_TICKER_URL,
        "fallbackRestMs": REST_FALLBACK_MS,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: #0b0f19;
      overflow: hidden;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .shell {{
      width: 100%;
      height: 760px;
      background: #0b0f19;
      border: 1px solid #1f2937;
      box-sizing: border-box;
    }}
    .header {{
      height: 40px;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 14px;
      color: #d1d5db;
      background: #0f172a;
      border-bottom: 1px solid #1f2937;
      font-size: 14px;
      white-space: nowrap;
    }}
    .symbol {{
      color: #f8fafc;
      font-weight: 700;
    }}
    .price {{
      color: #facc15;
      font-weight: 700;
    }}
    .action {{
      color: #93c5fd;
      font-weight: 600;
    }}
    .status {{
      margin-left: auto;
      color: #94a3b8;
      font-size: 12px;
    }}
    .warning {{
      display: none;
      position: absolute;
      right: 14px;
      top: 52px;
      max-width: 460px;
      padding: 7px 10px;
      color: #facc15;
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid rgba(250, 204, 21, 0.35);
      border-radius: 4px;
      font-size: 12px;
      z-index: 2;
    }}
    #chart {{
      width: 100%;
      height: 720px;
    }}
    #fallback {{
      display: none;
      color: #facc15;
      padding: 16px;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="header" id="header"></div>
    <div class="warning" id="ws-warning"></div>
    <div id="fallback"></div>
    <div id="chart"></div>
  </div>
  <script>
    const payload = {payload_json};
    const header = document.getElementById("header");
    const warningBox = document.getElementById("ws-warning");
    const fallback = document.getElementById("fallback");
    const container = document.getElementById("chart");

    let currentLastCandle = {{ ...payload.candles[payload.candles.length - 1] }};
    let previousLivePrice = Number.isFinite(payload.livePrice) ? Number(payload.livePrice) : currentLastCandle?.close;
    let livePrice = Number.isFinite(payload.livePrice) ? Number(payload.livePrice) : currentLastCandle?.close;
    let liveStatus = payload.liveEnabled ? "실시간 연결 중" : "실시간 꺼짐";
    let socket = null;
    let reconnectTimer = null;
    let restFallbackTimer = null;
    let pingTimer = null;
    let livePriceLine = null;

    function formatPrice(value) {{
      return Number.isFinite(value) ? Number(value).toLocaleString(undefined, {{
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }}) : "N/A";
    }}

    function renderHeader() {{
      const delta = Number.isFinite(livePrice) && Number.isFinite(previousLivePrice)
        ? livePrice - previousLivePrice
        : null;
      const deltaText = Number.isFinite(delta) ? `${{delta >= 0 ? "+" : ""}}${{formatPrice(delta)}}` : "N/A";
      header.innerHTML = `
        <span class="symbol">${{payload.symbol}}</span>
        <span>·</span>
        <span>${{payload.timeframe}}</span>
        <span>·</span>
        <span class="price">${{formatPrice(livePrice)}}</span>
        <span>·</span>
        <span>${{deltaText}}</span>
        <span>·</span>
        <span class="action">${{payload.actionHint}}</span>
        <span class="status">${{liveStatus}}</span>
      `;
    }}
    renderHeader();

    function showWarning(message) {{
      warningBox.style.display = "block";
      warningBox.textContent = message;
    }}

    function clearWarning() {{
      warningBox.style.display = "none";
      warningBox.textContent = "";
    }}

    function showFallback(message) {{
      fallback.style.display = "block";
      fallback.textContent = message;
      container.style.display = "none";
    }}

    if (!window.LightweightCharts) {{
      showFallback("lightweight-charts CDN failed to load.");
    }} else if (!payload.candles || payload.candles.length === 0) {{
      showFallback("No candle data available.");
    }} else {{
      const chart = LightweightCharts.createChart(container, {{
        width: container.clientWidth || 1200,
        height: 720,
        layout: {{
          background: {{ type: "solid", color: "#0b0f19" }},
          textColor: "#d1d5db",
        }},
        grid: {{
          vertLines: {{ color: "#1f2937" }},
          horzLines: {{ color: "#1f2937" }},
        }},
        crosshair: {{
          mode: LightweightCharts.CrosshairMode.Normal,
        }},
        rightPriceScale: {{
          visible: true,
          borderVisible: true,
          borderColor: "#374151",
          scaleMargins: {{ top: 0.08, bottom: 0.12 }},
        }},
        timeScale: {{
          visible: true,
          borderVisible: true,
          borderColor: "#374151",
          timeVisible: true,
          secondsVisible: payload.timeframe === "1m",
          rightOffset: 6,
          barSpacing: 8,
        }},
        handleScroll: {{
          mouseWheel: true,
          pressedMouseMove: true,
          horzTouchDrag: true,
          vertTouchDrag: true,
        }},
        handleScale: {{
          axisPressedMouseMove: true,
          mouseWheel: true,
          pinch: true,
        }},
      }});

      const candleSeries = chart.addCandlestickSeries({{
        upColor: "#26a69a",
        downColor: "#ef5350",
        wickUpColor: "#26a69a",
        wickDownColor: "#ef5350",
        borderUpColor: "#26a69a",
        borderDownColor: "#ef5350",
        priceLineVisible: true,
      }});
      candleSeries.setData(payload.candles);

      if (Number.isFinite(livePrice)) {{
        livePriceLine = candleSeries.createPriceLine({{
          price: Number(livePrice),
          color: "#facc15",
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Solid,
          axisLabelVisible: true,
          title: payload.livePriceLineTitle,
        }});
      }}

      const lineColors = {{
        ma_20: "rgba(250, 204, 21, 0.70)",
        ma_60: "rgba(96, 165, 250, 0.65)",
        ma_120: "rgba(192, 132, 252, 0.65)",
        bb_upper_20: "rgba(147, 197, 253, 0.45)",
        bb_lower_20: "rgba(147, 197, 253, 0.45)",
      }};

      Object.entries(payload.indicators || {{}}).forEach(([name, data]) => {{
        if (!data || data.length === 0) return;
        const line = chart.addLineSeries({{
          color: lineColors[name] || "rgba(209, 213, 219, 0.55)",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
          title: name,
        }});
        line.setData(data);
      }});

      if (Array.isArray(payload.markers) && payload.markers.length > 0) {{
        candleSeries.setMarkers(payload.markers);
      }}

      const savedRange = localStorage.getItem(payload.rangeKey);
      let restoredRange = false;
      if (savedRange) {{
        try {{
          const parsedRange = JSON.parse(savedRange);
          if (
            Number.isFinite(parsedRange.from) &&
            Number.isFinite(parsedRange.to) &&
            parsedRange.to > parsedRange.from
          ) {{
            chart.timeScale().setVisibleLogicalRange(parsedRange);
            restoredRange = true;
          }}
        }} catch (error) {{}}
      }}
      if (!restoredRange) {{
        chart.timeScale().fitContent();
      }}

      let rangeSaveTimer = null;
      chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {{
        if (!range) return;
        clearTimeout(rangeSaveTimer);
        rangeSaveTimer = setTimeout(() => {{
          localStorage.setItem(payload.rangeKey, JSON.stringify({{
            from: range.from,
            to: range.to,
          }}));
        }}, 120);
      }});

      const resize = () => {{
        chart.resize(container.clientWidth || 1200, 720);
      }};
      window.addEventListener("resize", resize);
      setTimeout(resize, 50);

      function updatePriceLine(nextPrice) {{
        if (livePriceLine) {{
          livePriceLine.applyOptions({{ price: nextPrice }});
          return;
        }}
        livePriceLine = candleSeries.createPriceLine({{
          price: nextPrice,
          color: "#facc15",
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Solid,
          axisLabelVisible: true,
          title: payload.livePriceLineTitle,
        }});
      }}

      function bucketStartFromMs(timestampMs) {{
        const eventSeconds = Math.floor(Number(timestampMs || Date.now()) / 1000);
        return Math.floor(eventSeconds / payload.bucketSeconds) * payload.bucketSeconds;
      }}

      function updateFromPrice(nextPrice, timestampMs, sourceLabel) {{
        if (!Number.isFinite(nextPrice) || !currentLastCandle) return;

        previousLivePrice = livePrice;
        livePrice = nextPrice;
        liveStatus = sourceLabel;

        const bucketStart = bucketStartFromMs(timestampMs);
        if (bucketStart <= Number(currentLastCandle.time)) {{
          currentLastCandle = {{
            ...currentLastCandle,
            close: nextPrice,
            high: Math.max(Number(currentLastCandle.high), nextPrice),
            low: Math.min(Number(currentLastCandle.low), nextPrice),
          }};
        }} else {{
          currentLastCandle = {{
            time: bucketStart,
            open: Number(currentLastCandle.close),
            high: nextPrice,
            low: nextPrice,
            close: nextPrice,
          }};
        }}

        candleSeries.update(currentLastCandle);
        updatePriceLine(nextPrice);
        renderHeader();
      }}

      async function pollRestFallback() {{
        try {{
          const response = await fetch(payload.restTickerUrl, {{ cache: "no-store" }});
          const data = await response.json();
          const nextPrice = Number(data?.result?.list?.[0]?.lastPrice);
          if (!Number.isFinite(nextPrice)) return;
          const timestamp = Number(data?.time) || Date.now();
          updateFromPrice(nextPrice, timestamp, "REST fallback");
        }} catch (error) {{
          showWarning(`REST fallback failed: ${{error?.message || error}}`);
          console.warn("REST fallback failed", error);
        }}
      }}

      function startRestFallback(reason) {{
        if (reason) showWarning(reason);
        if (restFallbackTimer) return;
        liveStatus = "REST 보조 조회";
        renderHeader();
        pollRestFallback();
        restFallbackTimer = window.setInterval(pollRestFallback, payload.fallbackRestMs);
      }}

      function stopRestFallback() {{
        if (!restFallbackTimer) return;
        window.clearInterval(restFallbackTimer);
        restFallbackTimer = null;
      }}

      function stopPing() {{
        if (!pingTimer) return;
        window.clearInterval(pingTimer);
        pingTimer = null;
      }}

      function scheduleReconnect() {{
        if (!payload.liveEnabled || reconnectTimer) return;
        reconnectTimer = window.setTimeout(() => {{
          reconnectTimer = null;
          connectWebSocket();
        }}, 5000);
      }}

      function priceFromTickerMessage(message) {{
        const data = Array.isArray(message?.data) ? message.data[0] : message?.data;
        const price = Number(data?.lastPrice || data?.price || data?.p);
        const timestamp = Number(data?.ts || message?.ts || Date.now());
        return {{ price, timestamp }};
      }}

      function connectWebSocket() {{
        if (!payload.liveEnabled) {{
          startRestFallback("실시간 가격 반영이 꺼져 있어 5초마다 REST로 보조 조회합니다.");
          return;
        }}

        try {{
          socket = new WebSocket(payload.wsUrl);
        }} catch (error) {{
          startRestFallback(`WebSocket 연결 생성에 실패했습니다. REST 보조 조회를 사용합니다. ${{error?.message || error}}`);
          scheduleReconnect();
          return;
        }}

        socket.onopen = () => {{
          clearWarning();
          stopRestFallback();
          stopPing();
          liveStatus = "실시간 가격 반영 중";
          renderHeader();
          socket.send(JSON.stringify({{ op: "subscribe", args: [payload.wsTopic] }}));
          pingTimer = window.setInterval(() => {{
            if (socket && socket.readyState === WebSocket.OPEN) {{
              socket.send(JSON.stringify({{ op: "ping" }}));
            }}
          }}, 20000);
        }};

        socket.onmessage = (event) => {{
          try {{
            const message = JSON.parse(event.data);
            if (message?.success === false) {{
              showWarning(`WebSocket 구독에 실패했습니다: ${{message?.ret_msg || "unknown error"}}`);
              startRestFallback("WebSocket 구독에 실패해 REST 보조 조회를 사용합니다.");
              return;
            }}
            const parsed = priceFromTickerMessage(message);
            if (Number.isFinite(parsed.price)) {{
              clearWarning();
              updateFromPrice(parsed.price, parsed.timestamp, "실시간 가격 반영 중");
            }}
          }} catch (error) {{
            console.warn("WebSocket message parse failed", error);
          }}
        }};

        socket.onerror = () => {{
          startRestFallback("WebSocket 오류가 발생해 REST 보조 조회를 사용합니다.");
        }};

        socket.onclose = () => {{
          stopPing();
          startRestFallback("WebSocket 연결이 끊겨 재연결하는 동안 REST 보조 조회를 사용합니다.");
          scheduleReconnect();
        }};
      }}

      connectWebSocket();

      window.addEventListener("beforeunload", () => {{
        if (socket) socket.close();
        if (restFallbackTimer) window.clearInterval(restFallbackTimer);
        if (reconnectTimer) window.clearTimeout(reconnectTimer);
        if (pingTimer) window.clearInterval(pingTimer);
      }});
    }}
  </script>
</body>
</html>
"""


def load_latest_row(path: Path, label: str, limit: int = 5) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, f"Missing {label}: {path}"
    try:
        df = normalize_timestamp(read_csv_tail(str(path), limit))
        return df, None
    except Exception as exc:
        return None, f"Failed to read {label}: {exc}"


def find_probability_column(df: pd.DataFrame, tokens: list[str]) -> str | None:
    if "pred_proba_drop" in df.columns:
        return "pred_proba_drop"
    if "pred_proba_high_drop" in df.columns:
        return "pred_proba_high_drop"
    for column in df.columns:
        lower = column.lower()
        if column != "timestamp" and "proba" in lower and all(token in lower for token in tokens):
            return column
    for column in df.columns:
        lower = column.lower()
        if column != "timestamp" and "proba" in lower and "low" not in lower:
            return column
    return None


def load_overlay_source(
    path: Path,
    label: str,
    columns: list[str],
    limit: int = RISK_OVERLAY_CANDLE_COUNT + 500,
) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, f"Missing {label}: {path}"
    try:
        df = normalize_timestamp(read_csv_tail(str(path), limit))
    except Exception as exc:
        return None, f"Failed to read {label}: {exc}"
    missing = [column for column in columns if column not in df.columns]
    if missing:
        return None, f"{label} is missing columns: {', '.join(missing)}"
    df = numeric_clean(df.loc[:, ["timestamp", *columns]], columns)
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True), None


def merge_overlay_source(
    base: pd.DataFrame,
    source: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    if source.empty:
        return base
    merged = pd.merge_asof(
        base.sort_values("timestamp"),
        source.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("15min"),
    )
    for column in columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged


def load_15m_risk_overlay_data(
    candles: pd.DataFrame,
) -> tuple[pd.DataFrame | None, list[str]]:
    warnings: list[str] = []
    overlay = candles[["timestamp", "open", "high", "low", "close"]].copy().sort_values("timestamp")

    vol_df, warning = load_overlay_source(
        VOLATILITY_PREDICTION_PATH,
        "high-vol predictions with indicators",
        ["pred_proba_high_vol"],
    )
    if warning:
        warnings.append(warning)
    elif vol_df is not None:
        overlay = merge_overlay_source(overlay, vol_df, ["pred_proba_high_vol"])

    drop_base = None
    if not DROP_RISK_PREDICTION_PATH.exists():
        warnings.append(f"Missing drop-risk predictions: {DROP_RISK_PREDICTION_PATH}")
    else:
        try:
            drop_base = normalize_timestamp(read_csv_tail(str(DROP_RISK_PREDICTION_PATH), RISK_OVERLAY_CANDLE_COUNT + 500))
        except Exception as exc:
            warnings.append(f"Failed to read drop-risk predictions: {exc}")
    if drop_base is not None:
        drop_column = find_probability_column(drop_base, ["drop"])
        if not drop_column:
            warnings.append("drop-risk predictions has no usable probability column.")
        else:
            drop_df = numeric_clean(drop_base.loc[:, ["timestamp", drop_column]], [drop_column])
            drop_df = drop_df.rename(columns={drop_column: "pred_proba_drop"})
            overlay = merge_overlay_source(overlay, drop_df, ["pred_proba_drop"])

    feature_df, warning = load_overlay_source(
        FEATURE_PATH,
        "features with indicators",
        ["atr_ratio_14", "bb_width_20"],
    )
    if warning:
        warnings.append(warning)
    elif feature_df is not None:
        overlay = merge_overlay_source(overlay, feature_df, ["atr_ratio_14", "bb_width_20"])

    for column in ["pred_proba_high_vol", "pred_proba_drop", "atr_ratio_14", "bb_width_20"]:
        if column not in overlay.columns:
            overlay[column] = pd.NA
        overlay[column] = pd.to_numeric(overlay[column], errors="coerce")

    return overlay.reset_index(drop=True), warnings


def load_1m_micro_candles(limit: int = MICRO_VIEW_CANDLE_COUNT) -> tuple[pd.DataFrame | None, str | None]:
    path = TIMEFRAME_PATHS[MICRO_VIEW_TIMEFRAME]
    if not path.exists():
        return None, f"Missing 1m candle file: {path}"
    try:
        df = normalize_timestamp(read_csv_tail(str(path), limit))
    except Exception as exc:
        return None, f"Failed to read 1m candle file: {exc}"

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        return None, f"1m candle file is missing columns: {', '.join(missing)}"

    df = numeric_clean(df.loc[:, required], ["open", "high", "low", "close", "volume"])
    df = df.dropna(subset=required)
    if df.empty:
        return None, "1m candle file has no usable OHLCV rows."
    return df.reset_index(drop=True), None


def build_micro_features(candles: pd.DataFrame) -> pd.DataFrame:
    df = candles.copy().sort_values("timestamp").reset_index(drop=True)
    df["return_1m"] = df["close"].pct_change()
    df["return_5m"] = df["close"] / df["close"].shift(5) - 1.0
    df["return_15m"] = df["close"] / df["close"].shift(15) - 1.0
    df["range_1m"] = (df["high"] - df["low"]) / df["close"]
    df["volume_ma_60"] = df["volume"].rolling(60, min_periods=20).mean()
    df["volume_ratio_60"] = df["volume"] / df["volume_ma_60"]
    df["volatility_20"] = df["return_1m"].rolling(20, min_periods=10).std()
    df["volatility_60"] = df["return_1m"].rolling(60, min_periods=20).std()
    df["ma_20"] = df["close"].rolling(20, min_periods=5).mean()
    df["ma_60"] = df["close"].rolling(60, min_periods=20).mean()
    df["range_ma_60"] = df["range_1m"].rolling(60, min_periods=20).mean()
    df["range_ratio_60"] = df["range_1m"] / df["range_ma_60"]
    return df.replace([float("inf"), float("-inf")], pd.NA)


def micro_event_state(row: pd.Series, controls: dict[str, Any], volatility_threshold: float, range_threshold: float) -> tuple[str, str]:
    return_5m = row.get("return_5m")
    volatility_20 = row.get("volatility_20")
    volume_ratio = row.get("volume_ratio_60")

    if pd.notna(return_5m) and float(return_5m) <= controls["fast_drop_5m_threshold"]:
        return "FAST_DROP", f"return_5m <= {controls['fast_drop_5m_threshold']:.4f}"
    if pd.notna(volatility_20) and float(volatility_20) >= volatility_threshold:
        return "VOL_SPIKE", f"volatility_20 >= p{controls['volatility_spike_percentile']:.2f}"
    if pd.notna(volume_ratio) and float(volume_ratio) >= controls["volume_spike_ratio_threshold"]:
        return "VOLUME_SPIKE", f"volume_ratio_60 >= {controls['volume_spike_ratio_threshold']:.2f}"
    if pd.notna(return_5m) and float(return_5m) >= controls["fast_pump_5m_threshold"]:
        return "FAST_PUMP", f"return_5m >= {controls['fast_pump_5m_threshold']:.4f}"

    range_1m = row.get("range_1m")
    range_ratio = row.get("range_ratio_60")
    if (
        pd.notna(range_1m)
        and pd.notna(range_ratio)
        and (float(range_ratio) >= controls["range_spike_ratio_threshold"] or float(range_1m) >= range_threshold)
    ):
        return "RANGE_SPIKE", "range spike"
    return "NORMAL", ""


def micro_marker_for_state(row: pd.Series, state: str) -> dict[str, Any] | None:
    text_by_state = {
        "FAST_DROP": display_event("FAST_DROP"),
        "FAST_PUMP": display_event("FAST_PUMP"),
        "VOLUME_SPIKE": display_event("VOLUME_SPIKE"),
        "VOL_SPIKE": display_event("VOL_SPIKE"),
    }
    color_by_state = {
        "FAST_DROP": "#fb7185",
        "FAST_PUMP": "#34d399",
        "VOLUME_SPIKE": "#facc15",
        "VOL_SPIKE": "#60a5fa",
    }
    if state not in text_by_state:
        return None
    return {
        "time": int(pd.Timestamp(row["timestamp"]).timestamp()),
        "position": "belowBar" if state == "FAST_DROP" else "aboveBar",
        "color": color_by_state[state],
        "shape": "arrowUp" if state in {"FAST_DROP", "FAST_PUMP"} else "circle",
        "text": text_by_state[state],
    }


def micro_event_row(row: pd.Series, state: str, reason: str) -> dict[str, Any]:
    return {
        "timestamp": row["timestamp"],
        "event": state,
        "close": None if pd.isna(row.get("close")) else float(row.get("close")),
        "return_1m": None if pd.isna(row.get("return_1m")) else float(row.get("return_1m")),
        "return_5m": None if pd.isna(row.get("return_5m")) else float(row.get("return_5m")),
        "return_15m": None if pd.isna(row.get("return_15m")) else float(row.get("return_15m")),
        "range_1m": None if pd.isna(row.get("range_1m")) else float(row.get("range_1m")),
        "volume_ratio_60": None if pd.isna(row.get("volume_ratio_60")) else float(row.get("volume_ratio_60")),
        "volatility_20": None if pd.isna(row.get("volatility_20")) else float(row.get("volatility_20")),
        "reason": reason,
    }


def build_micro_markers(
    features: pd.DataFrame,
    controls: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame, int, int, float, float]:
    clean_vol = pd.to_numeric(features["volatility_20"], errors="coerce").dropna()
    volatility_threshold = 0.0 if clean_vol.empty else float(clean_vol.quantile(controls["volatility_spike_percentile"]))
    clean_range = pd.to_numeric(features["range_1m"], errors="coerce").dropna()
    range_threshold = 0.0 if clean_range.empty else float(clean_range.quantile(0.95))

    markers: list[dict[str, Any]] = []
    marker_events: list[dict[str, Any]] = []
    raw_event_count = 0
    previous_state = "NORMAL"

    for _, row in features.iterrows():
        state, reason = micro_event_state(row, controls, volatility_threshold, range_threshold)
        if state != "NORMAL":
            raw_event_count += 1

        is_state_start = state != "NORMAL" and state != previous_state
        if is_state_start:
            show_marker = (
                (state == "FAST_DROP" and controls["show_fast_drop_markers"])
                or (state == "VOL_SPIKE" and controls["show_volatility_spike_markers"])
                or (state == "VOLUME_SPIKE" and controls["show_volume_spike_markers"])
                or (state == "FAST_PUMP" and controls["show_fast_pump_markers"])
            )
            event = micro_event_row(row, state, reason)
            if show_marker:
                marker = micro_marker_for_state(row, state)
                if marker:
                    markers.append(marker)
            marker_events.append(event)
        previous_state = state

    event_df = pd.DataFrame(marker_events)
    if not event_df.empty:
        event_df = event_df.sort_values("timestamp").tail(100).reset_index(drop=True)
    return markers, event_df, raw_event_count, len(markers), volatility_threshold, range_threshold


def risk_overlay_state(row: pd.Series, controls: dict[str, Any]) -> tuple[str, str]:
    high_vol = row.get("pred_proba_high_vol")
    drop = row.get("pred_proba_drop")
    high_vol_value = None if pd.isna(high_vol) else float(high_vol)
    drop_value = None if pd.isna(drop) else float(drop)

    reasons = []
    no_trade_high_vol = (
        high_vol_value is not None and high_vol_value >= controls["no_trade_high_vol_threshold"]
    )
    no_trade_drop = (
        drop_value is not None and drop_value >= controls["no_trade_drop_risk_threshold"]
    )
    if no_trade_high_vol:
        reasons.append(f"high-vol >= {controls['no_trade_high_vol_threshold']:.2f}")
    if no_trade_drop:
        reasons.append(f"drop-risk >= {controls['no_trade_drop_risk_threshold']:.2f}")
    if no_trade_high_vol or no_trade_drop:
        return "NO_TRADE", ", ".join(reasons)

    drop_hit = drop_value is not None and drop_value >= controls["drop_risk_marker_threshold"]
    if drop_hit:
        return "DROP_RISK", f"drop-risk >= {controls['drop_risk_marker_threshold']:.2f}"

    high_vol_hit = high_vol_value is not None and high_vol_value >= controls["high_vol_marker_threshold"]
    if high_vol_hit:
        return "HIGH_VOL", f"high-vol >= {controls['high_vol_marker_threshold']:.2f}"

    return "NORMAL", ""


def marker_for_state(row: pd.Series, state: str) -> dict[str, Any] | None:
    if state == "NO_TRADE":
        return {
            "time": int(pd.Timestamp(row["timestamp"]).timestamp()),
            "position": "aboveBar",
            "color": "#facc15",
            "shape": "circle",
            "text": display_event("NO_TRADE"),
        }
    if state == "DROP_RISK":
        marker = {
            "time": int(pd.Timestamp(row["timestamp"]).timestamp()),
            "position": "belowBar",
            "color": "#fb7185",
            "shape": "arrowUp",
            "text": display_event("DROP_RISK"),
        }
        return marker
    if state == "HIGH_VOL":
        return {
            "time": int(pd.Timestamp(row["timestamp"]).timestamp()),
            "position": "aboveBar",
            "color": "#60a5fa",
            "shape": "arrowDown",
            "text": display_event("HIGH_VOL"),
        }
    return None


def event_row_for_state(row: pd.Series, state: str, reason: str) -> dict[str, Any]:
    event_row = {
        "timestamp": row["timestamp"],
        "event": state,
        "pred_proba_high_vol": None if pd.isna(row.get("pred_proba_high_vol")) else float(row.get("pred_proba_high_vol")),
        "pred_proba_drop": None if pd.isna(row.get("pred_proba_drop")) else float(row.get("pred_proba_drop")),
        "atr_ratio_14": None if pd.isna(row.get("atr_ratio_14")) else float(row.get("atr_ratio_14")),
        "bb_width_20": None if pd.isna(row.get("bb_width_20")) else float(row.get("bb_width_20")),
        "reason": reason,
    }
    return event_row


def build_risk_overlay_markers(
    overlay: pd.DataFrame,
    controls: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame, int, int]:
    markers: list[dict[str, Any]] = []
    marker_events: list[dict[str, Any]] = []
    raw_events: list[dict[str, Any]] = []
    previous_state = "NORMAL"

    for _, row in overlay.iterrows():
        state, reason = risk_overlay_state(row, controls)
        if state != "NORMAL":
            raw_events.append(event_row_for_state(row, state, reason))

        is_state_start = state != "NORMAL" and state != previous_state
        if is_state_start:
            show_marker = (
                (state == "NO_TRADE" and controls["show_no_trade_markers"])
                or (state == "DROP_RISK" and controls["show_drop_risk_markers"])
                or (state == "HIGH_VOL" and controls["show_high_vol_markers"])
            )
            if show_marker:
                marker = marker_for_state(row, state)
                if marker:
                    markers.append(marker)
                    marker_events.append(event_row_for_state(row, state, reason))

        previous_state = state

    events = marker_events if controls["show_marker_events_only_in_table"] else raw_events
    event_df = pd.DataFrame(events)
    if not event_df.empty:
        event_df = event_df.sort_values("timestamp").tail(100).reset_index(drop=True)
    return markers, event_df, len(raw_events), len(markers)


def latest_similarity_summary() -> tuple[Path | None, dict[str, Any] | None, str | None]:
    if not SIMILARITY_DIR.exists():
        return None, None, f"Missing similarity directory: {SIMILARITY_DIR}"
    candidates = sorted(
        list(SIMILARITY_DIR.glob("*latest*summary*.json")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            return path, json.loads(path.read_text(encoding="utf-8")), None
        except Exception:
            continue
    return None, None, "Missing latest similarity summary JSON."


def risk_context() -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    context: dict[str, Any] = {
        "high_vol": None,
        "drop_risk": None,
        "atr_ratio_14": None,
        "bb_width_20": None,
        "action_hint": "NORMAL",
    }

    vol_df, warning = load_latest_row(VOLATILITY_PREDICTION_PATH, "high-vol predictions")
    if warning:
        warnings.append(warning)
    elif vol_df is not None and not vol_df.empty and "pred_proba_high_vol" in vol_df.columns:
        context["high_vol"] = pd.to_numeric(vol_df.iloc[-1]["pred_proba_high_vol"], errors="coerce")

    feature_df, warning = load_latest_row(FEATURE_PATH, "features with indicators")
    if warning:
        warnings.append(warning)
    elif feature_df is not None and not feature_df.empty:
        latest = feature_df.iloc[-1]
        for column in ["atr_ratio_14", "bb_width_20"]:
            if column in latest:
                context[column] = pd.to_numeric(latest[column], errors="coerce")

    drop_df, warning = load_latest_row(DROP_RISK_PREDICTION_PATH, "drop-risk predictions")
    if warning:
        warnings.append(warning)
    elif drop_df is not None and not drop_df.empty:
        drop_column = find_probability_column(drop_df, ["drop"])
        if drop_column:
            context["drop_risk"] = pd.to_numeric(drop_df.iloc[-1][drop_column], errors="coerce")

    _, _, similarity_warning = latest_similarity_summary()
    if similarity_warning:
        warnings.append(similarity_warning)

    context["action_hint"] = decide_action(context["high_vol"], context["drop_risk"])
    return context, warnings


def decide_action(high_vol: Any, drop_risk: Any) -> str:
    high_vol_value = None if high_vol is None or pd.isna(high_vol) else float(high_vol)
    drop_risk_value = None if drop_risk is None or pd.isna(drop_risk) else float(drop_risk)
    if drop_risk_value is not None and drop_risk_value >= 0.15:
        return "NO_TRADE"
    if high_vol_value is not None and high_vol_value >= 0.70:
        return "NO_TRADE"
    if high_vol_value is not None and high_vol_value >= 0.50:
        return "CAUTION"
    return "NORMAL"


def format_number(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        value_float = float(value)
        if pd.isna(value_float) or value_float in {float("inf"), float("-inf")}:
            return "-"
        return f"{value_float * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def format_ratio(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.{digits}f}x"
    except (TypeError, ValueError):
        return str(value)


@st.cache_data(show_spinner=False, ttl=5)
def fetch_rest_live_price() -> tuple[float | None, str | None]:
    try:
        with urllib.request.urlopen(BYBIT_REST_TICKER_URL, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return None, f"live price fetch failed: {exc}"

    try:
        ticker = payload["result"]["list"][0]
        return float(ticker["lastPrice"]), None
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return None, f"live price response has no usable lastPrice: {exc}"


def resolve_current_price(latest_close: Any, live_enabled: bool) -> tuple[float | None, str, str | None]:
    close_value = None if latest_close is None or pd.isna(latest_close) else float(latest_close)
    if live_enabled:
        live_price, warning = fetch_rest_live_price()
        if live_price is not None:
            return live_price, "bybit_rest_live", warning
        return close_value, "latest_1m_close", warning
    return close_value, "latest_1m_close", None


def current_price_gap_warning(current_price: float | None, latest_close: float | None) -> str | None:
    if current_price is None or latest_close is None or latest_close == 0:
        return None
    gap = abs(current_price - latest_close) / abs(latest_close)
    if gap >= 0.005:
        return (
            "현재가와 최신 1분봉 종가가 0.5% 이상 차이납니다. "
            f"current_price={current_price:.2f}, latest_1m_close={latest_close:.2f}, gap={gap * 100:.2f}%"
        )
    return None


def sidebar_data_status(extra_warnings: list[str]) -> None:
    with st.sidebar.expander(t("data_status"), expanded=False):
        for label, path in {
            "1m candles": TIMEFRAME_PATHS["1m"],
            "5m candles": TIMEFRAME_PATHS["5m"],
            "15m candles": TIMEFRAME_PATHS["15m"],
            "1h candles": TIMEFRAME_PATHS["1h"],
            "4h candles": TIMEFRAME_PATHS["4h"],
            "1d candles": TIMEFRAME_PATHS["1d"],
            "features": FEATURE_PATH,
            "high-vol predictions": VOLATILITY_PREDICTION_PATH,
            "drop-risk predictions": DROP_RISK_PREDICTION_PATH,
            "similarity directory": SIMILARITY_DIR,
        }.items():
            st.write(f"{'OK' if path.exists() else 'MISSING'} - `{label}`")
        for warning in extra_warnings:
            st.warning(warning)


def render_header_and_risk_cards(
    timeframe: str,
    risk: dict[str, Any],
) -> None:
    st.markdown(f"### BTCUSDT · {timeframe} · `{display_action_hint(risk['action_hint'])}`")
    st.caption(t("normal_caption"))

    cols = st.columns(5)
    cols[0].metric(
        t("high_vol_probability"),
        format_percent(risk["high_vol"]),
        help=t("high_vol_help"),
    )
    cols[1].metric(
        t("drop_risk_probability"),
        format_percent(risk["drop_risk"]),
        help=t("drop_risk_help"),
    )
    cols[2].metric(
        t("atr_ratio"),
        format_percent(risk["atr_ratio_14"]),
        help=t("atr_ratio_help"),
    )
    cols[3].metric(
        t("bb_width"),
        format_percent(risk["bb_width_20"]),
        help=t("bb_width_help"),
    )
    cols[4].metric(
        t("action_hint"),
        display_action_with_description(risk["action_hint"]),
        help=t("action_hint_help"),
    )


def render_market_chart(
    timeframe: str,
    recent_count: int,
    action_hint: str,
    live_enabled: bool,
    indicator_columns: list[str] | None = None,
    markers: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    candles, warning = load_ohlcv(timeframe, recent_count)
    if warning:
        st.error(warning)
        return None, None
    if candles is None:
        return None, None

    indicators = None
    if timeframe == "15m":
        indicators, indicator_warning = load_indicators_for_15m(recent_count, indicator_columns)
        if indicator_warning:
            st.sidebar.warning(indicator_warning)

    html = build_lightweight_chart_html(
        candles,
        indicators,
        timeframe,
        action_hint,
        live_enabled,
        indicator_columns=indicator_columns,
        markers=markers,
    )
    components.html(html, height=782, scrolling=False)

    sample = candles_to_chart_data(candles)[-1] if not candles.empty else None
    debug = {
        "loaded timeframe": timeframe,
        "loaded candle rows": len(candles),
        "first timestamp": str(candles["timestamp"].iloc[0]) if not candles.empty else "N/A",
        "last timestamp": str(candles["timestamp"].iloc[-1]) if not candles.empty else "N/A",
        "websocket live enabled": live_enabled,
        "websocket endpoint": BYBIT_WS_URL,
        "websocket topic": BYBIT_WS_TOPIC,
        "REST fallback interval ms": REST_FALLBACK_MS,
        "localStorage range key": f"btc_chart_range_{timeframe}_{len(candles_to_chart_data(candles))}",
        "risk overlay markers": 0 if markers is None else len(markers),
        "candle json sample": sample,
    }
    return candles, debug


def selected_indicator_columns(controls: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    if controls["show_ma_lines"]:
        columns.extend(MA_INDICATOR_COLUMNS)
    if controls["show_bollinger_bands"]:
        columns.extend(BB_INDICATOR_COLUMNS)
    return columns


def render_risk_overlay_controls() -> dict[str, Any]:
    st.sidebar.markdown(f"### {t('overlay_controls')}")
    return {
        "show_ma_lines": st.sidebar.checkbox(t("show_ma_lines"), value=True),
        "show_bollinger_bands": st.sidebar.checkbox(t("show_bollinger_bands"), value=False),
        "show_high_vol_markers": st.sidebar.checkbox(t("show_high_vol_markers"), value=False),
        "show_drop_risk_markers": st.sidebar.checkbox(t("show_drop_risk_markers"), value=False),
        "show_no_trade_markers": st.sidebar.checkbox(t("show_no_trade_markers"), value=True),
        "show_risk_event_table": st.sidebar.checkbox(t("show_risk_event_table"), value=True),
        "show_marker_events_only_in_table": st.sidebar.checkbox(t("show_marker_events_only_in_table"), value=True),
        "high_vol_marker_threshold": st.sidebar.slider(
            t("high_vol_marker_threshold"),
            min_value=0.50,
            max_value=0.95,
            value=DEFAULT_HIGH_VOL_MARKER_THRESHOLD,
            step=0.01,
        ),
        "drop_risk_marker_threshold": st.sidebar.slider(
            t("drop_risk_marker_threshold"),
            min_value=0.05,
            max_value=0.50,
            value=DEFAULT_DROP_RISK_MARKER_THRESHOLD,
            step=0.01,
        ),
        "no_trade_high_vol_threshold": st.sidebar.slider(
            t("no_trade_high_vol_threshold"),
            min_value=0.50,
            max_value=0.95,
            value=DEFAULT_NO_TRADE_HIGH_VOL_THRESHOLD,
            step=0.01,
        ),
        "no_trade_drop_risk_threshold": st.sidebar.slider(
            t("no_trade_drop_risk_threshold"),
            min_value=0.05,
            max_value=0.50,
            value=DEFAULT_NO_TRADE_DROP_RISK_THRESHOLD,
            step=0.01,
        ),
    }


def render_micro_overlay_controls() -> dict[str, Any]:
    st.sidebar.markdown(f"### {t('micro_overlay_controls')}")
    return {
        "show_ma_lines": st.sidebar.checkbox(t("show_ma_lines"), value=True),
        "show_volatility_spike_markers": st.sidebar.checkbox(t("show_volatility_spike_markers"), value=True),
        "show_volume_spike_markers": st.sidebar.checkbox(t("show_volume_spike_markers"), value=False),
        "show_fast_drop_markers": st.sidebar.checkbox(t("show_fast_drop_markers"), value=True),
        "show_fast_pump_markers": st.sidebar.checkbox(t("show_fast_pump_markers"), value=False),
        "show_micro_event_table": st.sidebar.checkbox(t("show_micro_event_table"), value=True),
        "fast_drop_5m_threshold": st.sidebar.slider(
            t("fast_drop_5m_threshold"),
            min_value=-0.02,
            max_value=-0.001,
            value=DEFAULT_FAST_DROP_5M_THRESHOLD,
            step=0.001,
            format="%.3f",
        ),
        "fast_pump_5m_threshold": st.sidebar.slider(
            t("fast_pump_5m_threshold"),
            min_value=0.001,
            max_value=0.02,
            value=DEFAULT_FAST_PUMP_5M_THRESHOLD,
            step=0.001,
            format="%.3f",
        ),
        "volume_spike_ratio_threshold": st.sidebar.slider(
            t("volume_spike_ratio_threshold"),
            min_value=1.5,
            max_value=10.0,
            value=DEFAULT_VOLUME_SPIKE_RATIO_THRESHOLD,
            step=0.1,
        ),
        "range_spike_ratio_threshold": st.sidebar.slider(
            t("range_spike_ratio_threshold"),
            min_value=1.5,
            max_value=10.0,
            value=DEFAULT_RANGE_SPIKE_RATIO_THRESHOLD,
            step=0.1,
        ),
        "volatility_spike_percentile": st.sidebar.slider(
            t("volatility_spike_percentile"),
            min_value=0.70,
            max_value=0.99,
            value=DEFAULT_VOLATILITY_SPIKE_PERCENTILE,
            step=0.01,
        ),
    }


def render_micro_risk_cards(features: pd.DataFrame, risk: dict[str, Any], current_price: float | None) -> None:
    latest = features.iloc[-1]
    st.markdown(f"### {t('micro_view_title')} · `{display_action_hint(risk['action_hint'])}`")
    st.caption(t("micro_view_caption"))
    cols = st.columns(7)
    cols[0].metric(t("live_btcusdt_price"), format_number(current_price, 2))
    cols[1].metric(t("return_1m"), format_percent(latest.get("return_1m")))
    cols[2].metric(t("return_5m"), format_percent(latest.get("return_5m")))
    cols[3].metric(t("return_15m"), format_percent(latest.get("return_15m")))
    cols[4].metric(t("volume_ratio_60"), format_ratio(latest.get("volume_ratio_60")))
    cols[5].metric(t("volatility_20"), format_number(latest.get("volatility_20"), 6))
    cols[6].metric(t("action_15m"), display_action_hint(risk["action_hint"]))


def render_15m_risk_overlay(action_hint: str, live_enabled: bool, controls: dict[str, Any]) -> dict[str, Any] | None:
    st.markdown(f"### {t('risk_overlay_title')}")
    st.caption(t("risk_overlay_caption"))

    candles, warning = load_ohlcv(RISK_OVERLAY_TIMEFRAME, RISK_OVERLAY_CANDLE_COUNT)
    if warning:
        st.error(warning)
        return None
    if candles is None:
        return None

    overlay, overlay_warnings = load_15m_risk_overlay_data(candles)
    for warning in overlay_warnings:
        st.warning(warning)

    markers: list[dict[str, Any]] = []
    events = pd.DataFrame()
    raw_threshold_event_count = 0
    compressed_marker_count = 0
    if overlay is not None:
        markers, events, raw_threshold_event_count, compressed_marker_count = build_risk_overlay_markers(overlay, controls)

    indicator_columns = selected_indicator_columns(controls)
    indicators = None
    if indicator_columns:
        indicators, indicator_warning = load_indicators_for_15m(RISK_OVERLAY_CANDLE_COUNT, indicator_columns)
        if indicator_warning:
            st.warning(indicator_warning)

    html = build_lightweight_chart_html(
        candles,
        indicators,
        RISK_OVERLAY_TIMEFRAME,
        action_hint,
        live_enabled,
        indicator_columns=indicator_columns,
        markers=markers,
    )
    components.html(html, height=782, scrolling=False)

    if controls["show_risk_event_table"]:
        st.markdown(f"#### {t('recent_risk_events')}")
        if events.empty:
            st.info(t("no_risk_events"))
        else:
            table = display_table(
                events,
                [
                    "timestamp",
                    "event",
                    "pred_proba_high_vol",
                    "pred_proba_drop",
                    "atr_ratio_14",
                    "bb_width_20",
                    "reason",
                ],
                "risk_event_columns",
            )
            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
            )

    return {
        "loaded timeframe": RISK_OVERLAY_TIMEFRAME,
        "loaded candle rows": len(candles),
        "fixed recent candles": RISK_OVERLAY_CANDLE_COUNT,
        "first timestamp": str(candles["timestamp"].iloc[0]) if not candles.empty else "N/A",
        "last timestamp": str(candles["timestamp"].iloc[-1]) if not candles.empty else "N/A",
        "indicator columns": indicator_columns,
        "overlay merged rows": 0 if overlay is None else len(overlay),
        "raw threshold event count": raw_threshold_event_count,
        "compressed marker count": compressed_marker_count,
        "risk event rows": 0 if events.empty else len(events),
        "high-vol marker threshold": controls["high_vol_marker_threshold"],
        "drop-risk marker threshold": controls["drop_risk_marker_threshold"],
        "NO_TRADE high-vol threshold": controls["no_trade_high_vol_threshold"],
        "NO_TRADE drop-risk threshold": controls["no_trade_drop_risk_threshold"],
        "websocket live enabled": live_enabled,
    }


def render_1m_live_micro_view(risk: dict[str, Any], live_enabled: bool, controls: dict[str, Any]) -> dict[str, Any] | None:
    candles, warning = load_1m_micro_candles(MICRO_VIEW_CANDLE_COUNT)
    if warning:
        st.error(warning)
        return None
    if candles is None:
        return None

    features = build_micro_features(candles)
    latest_close = None if features.empty else float(features.iloc[-1]["close"])
    current_price, current_price_source, current_price_warning = resolve_current_price(latest_close, live_enabled)
    if current_price_warning:
        st.warning(current_price_warning)
    gap_warning = current_price_gap_warning(current_price, latest_close)
    if gap_warning:
        st.warning(gap_warning)
    render_micro_risk_cards(features, risk, current_price)

    markers, events, raw_event_count, compressed_marker_count, volatility_threshold, range_threshold = build_micro_markers(
        features, controls
    )
    indicator_columns = ["ma_20", "ma_60"] if controls["show_ma_lines"] else []
    indicators = features.loc[:, ["timestamp", *indicator_columns]].copy() if indicator_columns else None

    html = build_lightweight_chart_html(
        candles,
        indicators,
        MICRO_VIEW_TIMEFRAME,
        risk["action_hint"],
        live_enabled,
        indicator_columns=indicator_columns,
        markers=markers,
        current_price=current_price,
    )
    components.html(html, height=782, scrolling=False)

    if controls["show_micro_event_table"]:
        st.markdown(f"#### {t('recent_micro_events')}")
        if events.empty:
            st.info(t("no_micro_events"))
        else:
            table = display_table(
                events,
                [
                    "timestamp",
                    "event",
                    "close",
                    "return_1m",
                    "return_5m",
                    "return_15m",
                    "range_1m",
                    "volume_ratio_60",
                    "volatility_20",
                    "reason",
                ],
                "micro_event_columns",
            )
            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
            )

    return {
        "loaded timeframe": MICRO_VIEW_TIMEFRAME,
        "loaded 1m candles": len(candles),
        "micro feature rows": len(features),
        "first timestamp": str(candles["timestamp"].iloc[0]) if not candles.empty else "N/A",
        "last timestamp": str(candles["timestamp"].iloc[-1]) if not candles.empty else "N/A",
        "current_price": current_price,
        "current_price source": current_price_source,
        "latest 1m close": latest_close,
        "indicator columns": indicator_columns,
        "raw micro event count": raw_event_count,
        "compressed marker count": compressed_marker_count,
        "event table rows": 0 if events.empty else len(events),
        "volatility threshold": volatility_threshold,
        "range threshold": range_threshold,
        "websocket live enabled": live_enabled,
    }


def show_debug(debug: dict[str, Any] | None) -> None:
    with st.expander(t("chart_debug"), expanded=False):
        if not debug:
            st.write(t("no_chart_debug"))
            return
        for key, value in debug.items():
            st.write(f"**{key}:** `{value}`")


def show_baseline_comparison() -> None:
    with st.expander(t("baseline_title"), expanded=False):
        st.write(t("baseline_caption"))


def main() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 100%; }
        [data-testid="stMetricValue"] { font-size: 1.05rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    view_modes = ["Normal Dashboard", "15m Risk Overlay", "1m Live Micro View"]
    view_mode = st.sidebar.selectbox(
        t("view_mode"),
        view_modes,
        index=0,
        format_func=lambda value: t("view_modes").get(value, value),
    )
    live_enabled = st.sidebar.checkbox(t("websocket_live"), value=True)
    st.sidebar.caption(t("fallback_rest"))

    risk, risk_warnings = risk_context()
    sidebar_data_status(risk_warnings)

    if view_mode == "15m Risk Overlay":
        controls = render_risk_overlay_controls()
        render_header_and_risk_cards(RISK_OVERLAY_TIMEFRAME, risk)
        debug = render_15m_risk_overlay(risk["action_hint"], live_enabled, controls)
    elif view_mode == "1m Live Micro View":
        controls = render_micro_overlay_controls()
        debug = render_1m_live_micro_view(risk, live_enabled, controls)
    else:
        timeframe = st.sidebar.selectbox(t("timeframe"), ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)
        recent_count = st.sidebar.selectbox(t("recent_candles"), [100, 200, 300, 500, 1000], index=1)
        render_header_and_risk_cards(timeframe, risk)
        _, debug = render_market_chart(timeframe, recent_count, risk["action_hint"], live_enabled)

    show_debug(debug)
    show_baseline_comparison()


if __name__ == "__main__":
    main()

