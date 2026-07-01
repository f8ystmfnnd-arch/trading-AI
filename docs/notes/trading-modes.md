# Trading Modes

## Project Direction

BTC Market Regime & Risk Guard AI is not primarily a short-term price prediction project.

The core purpose is to detect market risk regimes and support risk management decisions:

- Identify dangerous market regimes before taking new risk.
- Use directional models only as auxiliary filters.
- Use high-volatility and sharp-drop risk models to decide entry blocks, position reduction, and strategy ON/OFF states.
- Separate fast intraday decisions from slower swing-position decisions.

In this structure, a model output is not automatically a buy or sell signal. It is an input into a risk guard.

## Day Trading Mode

### Goal

Day Trading Mode asks whether it is safe to enter or keep trading inside the current day.

It focuses on short-term risk over the next 15 minutes to 1 hour. The key concern is avoiding fee drag, market noise, high-volatility bursts, and sharp downside moves.

### Timeframes

- 1m
- 5m
- 15m
- 1h as auxiliary context

### Main Prediction Targets

- Next 15-minute direction
- Next 1-hour high-volatility probability
- Next 1-hour sharp-drop risk
- Next 1-hour large-move risk

Existing `next_4` targets belong to Day Trading Mode.

On a 15-minute base timeframe:

- `next_4` = next 4 candles
- `next_4` = next 1 hour

### Usage

Day Trading Mode can be used to decide:

- Allow new entries
- Warn on new entries
- Block new entries
- Reduce position size
- Trigger cooldown after consecutive losses
- Restrict trading during high-volatility periods

## Swing Trading Mode

### Goal

Swing Trading Mode asks whether the market is worth holding for several days.

It focuses on whether to hold, reduce, or open a position over a multi-day horizon. Compared with Day Trading Mode, it cares less about short-term noise and more about 4-hour / 1-day risk and market regime.

### Timeframes

- 15m
- 1h
- 4h
- 1d

### Main Prediction Targets

- Next 4-hour return, volatility, and sharp-drop risk
- Next 1-day return, volatility, and sharp-drop risk
- Market regime: uptrend, downtrend, range, high-volatility regime

On a 15-minute base timeframe:

- `next_16` = next 16 candles = next 4 hours
- `next_96` = next 96 candles = next 1 day

### Swing Target Candidates

- `target_return_next_16`
- `target_volatility_high_next_16`
- `target_drop_next_16`
- `target_return_next_96`
- `target_volatility_high_next_96`
- `target_drop_next_96`

### Usage

Swing Trading Mode can be used to decide:

- Whether swing entries are allowed
- Whether existing positions should be held
- Whether position size should be reduced
- Whether to stay out during high-risk regimes
- Whether to take defensive action in downtrend plus high-volatility regimes

## Mode Comparison

| Category | Day Trading Mode | Swing Trading Mode |
|---|---|---|
| Purpose | Decide whether short-term entry is allowed now | Decide whether the market is worth holding for several days |
| Main Time Unit | 15 minutes to 1 hour | 4 hours to 1 day |
| Main Risks | Fees, noise, sudden volatility, short-term drops | Regime shift, trend deterioration, multi-hour or daily drawdown |
| Main Timeframes | 1m, 5m, 15m, 1h auxiliary | 15m, 1h, 4h, 1d |
| Model Outputs | Next 15m direction, next 1h high volatility, next 1h drop risk, next 1h big move | Next 4h / 1d return, volatility, drop risk, market regime |
| Final Actions | Allow entry, caution, block entry, reduce size, cooldown | Allow swing entry, hold, reduce, avoid high-risk zone, defensive posture |

## One-Line Definition

Day Trading Mode asks: **Can we enter right now?**

Swing Trading Mode asks: **Is this market worth holding for several days?**
