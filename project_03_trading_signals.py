"""
============================================================
PROJECT 3: Algorithmic Trading Signal Generation (MACD + RSI)
============================================================
Industry Relevance (2025-2026):
  - Quant funds, prop trading desks, algo trading startups
  - Used at: Two Sigma, Citadel, Jane Street, Zerodha,
    Alpaca, Interactive Brokers integrations
  - Hiring: Quantitative Analyst | Algo Trader | ML Quant

Tech Stack:
  yfinance | MACD | RSI | Bollinger Bands | Backtesting |
  Matplotlib | pandas | Performance Metrics
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("  PROJECT 3: Algorithmic Trading Signal Generation")
print("  Indicators: MACD + RSI + Bollinger Bands + Backtesting")
print("=" * 65)

# ── 1. Download stock data ────────────────────────────────────────
TICKER  = "AAPL"
START   = "2019-01-01"
END     = "2024-01-01"

print(f"\n📡 Downloading {TICKER} data ({START} → {END})...")
data = yf.download(TICKER, start=START, end=END, progress=False)
if data.empty:
    print("⚠️  yfinance download failed — using simulated data")
    idx  = pd.date_range(START, END, freq="B")
    np.random.seed(42)
    data = pd.DataFrame({
        "Close":  100 + np.cumsum(np.random.normal(0.05, 1.2, len(idx))),
        "Volume": np.random.randint(5_000_000, 25_000_000, len(idx))
    }, index=idx)
    data["Open"] = data["Close"] * np.random.uniform(0.995, 1.005, len(data))
    data["High"] = data["Close"] * np.random.uniform(1.000, 1.020, len(data))
    data["Low"]  = data["Close"] * np.random.uniform(0.980, 1.000, len(data))
else:
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

close = data["Close"].squeeze()
print(f"✅  Loaded {len(close)} trading days")

# ── 2. Compute all technical indicators ──────────────────────────

# MACD (12-26-9)
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
macd_line   = ema12 - ema26
signal_line = macd_line.ewm(span=9, adjust=False).mean()
macd_hist   = macd_line - signal_line

# RSI (14-period)
delta = close.diff()
gain  = delta.where(delta > 0, 0).rolling(14).mean()
loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs    = gain / loss
rsi   = 100 - (100 / (1 + rs))

# Bollinger Bands (20-period)
bb_window = 20
bb_mid    = close.rolling(bb_window).mean()
bb_std    = close.rolling(bb_window).std()
bb_upper  = bb_mid + 2 * bb_std
bb_lower  = bb_mid - 2 * bb_std
bb_width  = (bb_upper - bb_lower) / bb_mid  # bandwidth

# ── 3. Generate composite trading signals ────────────────────────
# BUY: MACD crosses above signal AND RSI < 40 (not extreme overbought)
# SELL: MACD crosses below signal AND RSI > 60 (not extreme oversold)
macd_cross_up   = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
macd_cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

buy_signal  = macd_cross_up  & (rsi < 45)
sell_signal = macd_cross_down & (rsi > 55)

# Bollinger Band squeeze breakout signals
bb_squeeze  = bb_width < bb_width.rolling(50).mean() * 0.7
bb_breakout_up   = (close > bb_upper) & bb_squeeze.shift(5)
bb_breakout_down = (close < bb_lower) & bb_squeeze.shift(5)

signals = pd.DataFrame({
    "Close": close,
    "MACD": macd_line, "Signal_Line": signal_line, "MACD_Hist": macd_hist,
    "RSI": rsi,
    "BB_Mid": bb_mid, "BB_Upper": bb_upper, "BB_Lower": bb_lower,
    "Buy_Signal": buy_signal, "Sell_Signal": sell_signal,
    "BB_Breakout_Up": bb_breakout_up, "BB_Breakout_Down": bb_breakout_down,
})

n_buy  = buy_signal.sum()
n_sell = sell_signal.sum()
print(f"\n📈 Buy signals generated:  {n_buy}")
print(f"📉 Sell signals generated: {n_sell}")

# ── 4. Backtesting engine ────────────────────────────────────────
print("\n⚙️  Running backtest...")

cash          = 10_000.0
shares        = 0
initial_cash  = cash
trade_log     = []
portfolio_val = []

for date, row in signals.dropna().iterrows():
    price = float(row["Close"])
    # Execute BUY
    if row["Buy_Signal"] and cash >= price:
        shares = int(cash * 0.95 // price)   # Use 95% of capital
        cost   = shares * price
        cash  -= cost
        trade_log.append({"date": date, "action": "BUY",
                           "price": price, "shares": shares, "cash": cash})
    # Execute SELL
    elif row["Sell_Signal"] and shares > 0:
        revenue = shares * price
        cash   += revenue
        trade_log.append({"date": date, "action": "SELL",
                           "price": price, "shares": shares, "cash": cash})
        shares = 0
    portfolio_val.append(cash + shares * price)

# Close any remaining position
final_price = float(signals["Close"].iloc[-1])
if shares > 0:
    cash += shares * final_price

final_balance  = cash
total_return   = (final_balance - initial_cash) / initial_cash * 100
buy_hold_ret   = (final_price - float(signals["Close"].iloc[0])) / float(signals["Close"].iloc[0]) * 100

# Portfolio series aligned to signals index
port_series = pd.Series(portfolio_val, index=signals.dropna().index)

print(f"\n💰 Backtest Results:")
print(f"   Initial Capital:     ${initial_cash:>12,.2f}")
print(f"   Final Balance:       ${final_balance:>12,.2f}")
print(f"   Strategy Return:     {total_return:>+11.2f}%")
print(f"   Buy & Hold Return:   {buy_hold_ret:>+11.2f}%")
print(f"   Alpha vs B&H:        {total_return-buy_hold_ret:>+11.2f}%")
print(f"   Total Trades:        {len(trade_log):>12}")

# ── 5. Four-panel professional chart ────────────────────────────
DARK_BG = "#1a1d27"

def style_ax(ax, title, ylabel=""):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    if ylabel: ax.set_ylabel(ylabel, color="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#2d3142")
    ax.grid(True, alpha=0.15, color="white")

fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.08,
                        height_ratios=[3, 1, 1, 1.5])

s = signals.dropna()

# Panel 1: Price + Bollinger + Buy/Sell signals
ax1 = fig.add_subplot(gs[0])
ax1.plot(s.index, s["Close"],    color="#4f9cf9", lw=1.5, label="Price", zorder=3)
ax1.plot(s.index, s["BB_Upper"], color="#f59e0b", lw=0.8, linestyle="--", alpha=0.7, label="BB Upper")
ax1.plot(s.index, s["BB_Mid"],   color="#6b7280", lw=0.8, linestyle="--", alpha=0.5, label="BB Mid")
ax1.plot(s.index, s["BB_Lower"], color="#f59e0b", lw=0.8, linestyle="--", alpha=0.7, label="BB Lower")
ax1.fill_between(s.index, s["BB_Upper"], s["BB_Lower"], alpha=0.06, color="#f59e0b")
ax1.scatter(s.index[s["Buy_Signal"]], s["Close"][s["Buy_Signal"]],
            marker="^", color="#22c55e", s=80, zorder=5, label=f"Buy ({n_buy})")
ax1.scatter(s.index[s["Sell_Signal"]], s["Close"][s["Sell_Signal"]],
            marker="v", color="#ef4444", s=80, zorder=5, label=f"Sell ({n_sell})")
ax1.set_ylabel("Price (USD)", color="white")
ax1.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white",
           loc="upper left", ncol=4)
style_ax(ax1, f"{TICKER} Price + Bollinger Bands + Trading Signals")
ax1.set_xticklabels([])

# Panel 2: MACD
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(s.index, s["MACD"],        color="#4f9cf9", lw=1.2, label="MACD")
ax2.plot(s.index, s["Signal_Line"], color="#f59e0b", lw=1.0, label="Signal")
colors_hist = ["#22c55e" if v >= 0 else "#ef4444" for v in s["MACD_Hist"]]
ax2.bar(s.index, s["MACD_Hist"], color=colors_hist, alpha=0.7, width=1)
ax2.axhline(0, color="white", lw=0.5, alpha=0.4)
ax2.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")
style_ax(ax2, "MACD (12-26-9)", "MACD")
ax2.set_xticklabels([])

# Panel 3: RSI
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.plot(s.index, s["RSI"], color="#a78bfa", lw=1.2, label="RSI")
ax3.fill_between(s.index, 70, s["RSI"], where=s["RSI"]>70, alpha=0.3, color="#ef4444")
ax3.fill_between(s.index, 30, s["RSI"], where=s["RSI"]<30, alpha=0.3, color="#22c55e")
ax3.axhline(70, color="#ef4444", lw=0.8, linestyle="--", label="Overbought 70")
ax3.axhline(30, color="#22c55e", lw=0.8, linestyle="--", label="Oversold 30")
ax3.axhline(50, color="white",   lw=0.5, linestyle=":", alpha=0.4)
ax3.set_ylim(0, 100)
ax3.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")
style_ax(ax3, "RSI (14-period)", "RSI")
ax3.set_xticklabels([])

# Panel 4: Portfolio vs Buy & Hold
ax4 = fig.add_subplot(gs[3], sharex=ax1)
bh_series = (s["Close"] / float(s["Close"].iloc[0])) * initial_cash
ax4.plot(port_series.index, port_series.values, color="#22c55e",
         lw=1.5, label=f"Strategy (${final_balance:,.0f})")
ax4.plot(bh_series.index,   bh_series.values,   color="#f59e0b",
         lw=1.5, linestyle="--",
         label=f"Buy & Hold (${bh_series.iloc[-1]:,.0f})")
ax4.axhline(initial_cash, color="white", lw=0.5, linestyle=":", alpha=0.4)
ax4.set_ylabel("Portfolio USD", color="white")
ax4.legend(fontsize=8, facecolor=DARK_BG, labelcolor="white")
style_ax(ax4, "Portfolio Value: Strategy vs Buy & Hold", "Value USD")

fig.suptitle(f"Trading Signal Generation — {TICKER} Algorithmic Strategy",
             color="white", fontsize=15, fontweight="bold", y=1.005)
plt.savefig("outputs/03_trading_signals.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("\n📊 Chart → outputs/03_trading_signals.png")

# ── 6. Trade log summary ─────────────────────────────────────────
if trade_log:
    tl_df = pd.DataFrame(trade_log)
    buy_trades  = tl_df[tl_df["action"] == "BUY"]
    sell_trades = tl_df[tl_df["action"] == "SELL"]
    print(f"\n📑 Trade Log Summary (last 5):")
    print(tl_df.tail(5).to_string(index=False))

print("\n✅ Project 3 — Trading Signal Generation complete.\n")
