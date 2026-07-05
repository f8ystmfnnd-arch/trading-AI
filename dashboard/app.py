"""TradingView-style dashboard for BTC Market Regime & Risk Guard AI."""

from __future__ import annotations

import json
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
INDICATOR_COLUMNS = ["ma_20", "ma_60", "ma_120", "bb_upper_20", "bb_lower_20"]
TIMEFRAME_BUCKET_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


st.set_page_config(page_title="BTC Market Regime & Risk Guard AI", layout="wide")


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


def load_indicators_for_15m(limit: int) -> tuple[pd.DataFrame | None, str | None]:
    if not FEATURE_PATH.exists():
        return None, f"Missing 15m indicator file: {FEATURE_PATH}"
    try:
        df = normalize_timestamp(read_csv_tail(str(FEATURE_PATH), limit + 300))
    except Exception as exc:
        return None, f"Failed to read 15m indicator file: {exc}"

    columns = ["timestamp", *[column for column in INDICATOR_COLUMNS if column in df.columns]]
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


def indicators_to_chart_data(indicators: pd.DataFrame | None, candles: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if indicators is None or indicators.empty:
        return {}
    merged = candles[["timestamp"]].merge(indicators, on="timestamp", how="left")
    merged["time"] = unix_seconds(merged["timestamp"])

    output: dict[str, list[dict[str, Any]]] = {}
    for column in INDICATOR_COLUMNS:
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
) -> str:
    candle_data = candles_to_chart_data(candles)
    indicator_data = indicators_to_chart_data(indicators, candles)
    latest_close = float(candles.iloc[-1]["close"]) if not candles.empty else None

    payload = {
        "symbol": "BTCUSDT",
        "timeframe": timeframe,
        "candles": candle_data,
        "indicators": indicator_data,
        "livePrice": latest_close,
        "actionHint": action_hint,
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
    let liveStatus = payload.liveEnabled ? "WebSocket connecting" : "WebSocket OFF";
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
          title: "Live BTCUSDT",
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
          title: "Live BTCUSDT",
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
        liveStatus = "REST fallback";
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
          startRestFallback("WebSocket live is OFF. Using REST fallback every 5 seconds.");
          return;
        }}

        try {{
          socket = new WebSocket(payload.wsUrl);
        }} catch (error) {{
          startRestFallback(`WebSocket creation failed. REST fallback active. ${{error?.message || error}}`);
          scheduleReconnect();
          return;
        }}

        socket.onopen = () => {{
          clearWarning();
          stopRestFallback();
          stopPing();
          liveStatus = "WebSocket LIVE";
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
              showWarning(`WebSocket subscription failed: ${{message?.ret_msg || "unknown error"}}`);
              startRestFallback("WebSocket subscription failed. REST fallback active.");
              return;
            }}
            const parsed = priceFromTickerMessage(message);
            if (Number.isFinite(parsed.price)) {{
              clearWarning();
              updateFromPrice(parsed.price, parsed.timestamp, "WebSocket LIVE");
            }}
          }} catch (error) {{
            console.warn("WebSocket message parse failed", error);
          }}
        }};

        socket.onerror = () => {{
          startRestFallback("WebSocket error. REST fallback active.");
        }};

        socket.onclose = () => {{
          stopPing();
          startRestFallback("WebSocket disconnected. REST fallback active while reconnecting.");
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


def sidebar_data_status(extra_warnings: list[str]) -> None:
    with st.sidebar.expander("Data status", expanded=False):
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
        }.items():
            st.write(f"{'OK' if path.exists() else 'MISSING'} - `{label}`")
        for warning in extra_warnings:
            st.warning(warning)


def render_header_and_risk_cards(
    timeframe: str,
    risk: dict[str, Any],
) -> None:
    st.markdown(f"### BTCUSDT · {timeframe} · `{risk['action_hint']}`")
    st.caption(
        "Live updates are visual references. Risk Guard decisions are based on 15m features/models."
    )

    cols = st.columns(5)
    cols[0].metric("High-vol probability", format_number(risk["high_vol"]))
    cols[1].metric("Drop risk probability", format_number(risk["drop_risk"]))
    cols[2].metric("ATR ratio", format_number(risk["atr_ratio_14"], 6))
    cols[3].metric("BB width", format_number(risk["bb_width_20"], 6))
    cols[4].metric("Action hint", risk["action_hint"])


def render_market_chart(
    timeframe: str,
    recent_count: int,
    action_hint: str,
    live_enabled: bool,
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    candles, warning = load_ohlcv(timeframe, recent_count)
    if warning:
        st.error(warning)
        return None, None
    if candles is None:
        return None, None

    indicators = None
    if timeframe == "15m":
        indicators, indicator_warning = load_indicators_for_15m(recent_count)
        if indicator_warning:
            st.sidebar.warning(indicator_warning)

    html = build_lightweight_chart_html(candles, indicators, timeframe, action_hint, live_enabled)
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
        "candle json sample": sample,
    }
    return candles, debug


def show_debug(debug: dict[str, Any] | None) -> None:
    with st.expander("Chart debug", expanded=False):
        if not debug:
            st.write("No chart debug data available.")
            return
        for key, value in debug.items():
            st.write(f"**{key}:** `{value}`")


def show_baseline_comparison() -> None:
    with st.expander("Indicator model baseline comparison", expanded=False):
        st.write(
            """
            Technical indicators are used as market-regime features, not mechanical trading formulas.

            - `roc_auc`: `0.788064` -> `0.795049`
            - `average_precision`: `0.599165` -> `0.610102`
            - `log_loss`: `0.525593` -> `0.512464`
            """
        )


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

    timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)
    recent_count = st.sidebar.selectbox("Recent candles", [100, 200, 300, 500, 1000], index=1)
    live_enabled = st.sidebar.checkbox("WebSocket live", value=True)
    st.sidebar.caption("Fallback REST polling interval: 5s")

    risk, risk_warnings = risk_context()
    sidebar_data_status(risk_warnings)

    render_header_and_risk_cards(timeframe, risk)
    _, debug = render_market_chart(timeframe, recent_count, risk["action_hint"], live_enabled)
    show_debug(debug)
    show_baseline_comparison()


if __name__ == "__main__":
    main()

