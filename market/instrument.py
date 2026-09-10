"""Instrument identity shared by historical collection and live display."""

BYBIT_EXCHANGE = "bybit"
BYBIT_CATEGORY = "spot"
BYBIT_SYMBOL = "BTCUSDT"
BYBIT_REST_TICKER_URL = (
    f"https://api.bybit.com/v5/market/tickers?category={BYBIT_CATEGORY}&symbol={BYBIT_SYMBOL}"
)
BYBIT_WS_URL = f"wss://stream.bybit.com/v5/public/{BYBIT_CATEGORY}"
