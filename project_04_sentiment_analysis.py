"""
============================================================
PROJECT 4: Financial News Sentiment Analysis (VADER + NLP)
============================================================
Industry Relevance (2025-2026):
  - Hedge funds use news sentiment as alpha signals
  - ESG scoring platforms, news-driven quant strategies
  - Used at: Bloomberg Terminal, Refinitiv, Two Sigma,
    AQR Capital, Man Group, Kensho (S&P)
  - Hiring: NLP Engineer | Quant Researcher | Data Scientist

Tech Stack:
  VADER | TextBlob | NLTK | pandas | Matplotlib |
  Sentiment Pipeline | Trend Analysis | Signal Generation
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from textblob import TextBlob
import re
import warnings
warnings.filterwarnings("ignore")

# VADER
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download("vader_lexicon", quiet=True)
    VADER_OK = True
except Exception:
    VADER_OK = False

print("=" * 65)
print("  PROJECT 4: Financial News Sentiment Analysis")
print("=" * 65)

# ── 1. Rich financial news dataset ───────────────────────────────
headlines_data = [
    # (date, company, headline)
    ("2024-01-02", "AAPL", "Apple reports record Q4 earnings, beats revenue estimates by 8%"),
    ("2024-01-03", "AAPL", "Apple faces antitrust investigation in EU over App Store practices"),
    ("2024-01-04", "AAPL", "Apple Vision Pro pre-orders surpass analyst expectations"),
    ("2024-01-05", "AAPL", "Apple supplier warns of demand slowdown in China market"),
    ("2024-01-08", "AAPL", "Apple announces $110 billion share buyback program"),
    ("2024-01-09", "MSFT", "Microsoft AI revenue grows 35% driven by Azure OpenAI adoption"),
    ("2024-01-10", "MSFT", "Microsoft Teams faces regulatory scrutiny in Europe"),
    ("2024-01-11", "MSFT", "Microsoft Copilot integration boosts enterprise subscriptions"),
    ("2024-01-12", "MSFT", "Microsoft cloud division misses quarterly growth target"),
    ("2024-01-15", "MSFT", "Microsoft announces 1900 gaming division layoffs post-Activision"),
    ("2024-01-16", "GOOGL","Google DeepMind achieves breakthrough in protein structure prediction"),
    ("2024-01-17", "GOOGL","Google faces $4B fine for search monopoly from DOJ"),
    ("2024-01-18", "GOOGL","Google Cloud revenue accelerates, market share hits 12%"),
    ("2024-01-19", "GOOGL","Alphabet beats EPS estimate; ad revenue recovers strongly"),
    ("2024-01-22", "GOOGL","Google layoffs hit 12,000 employees across core divisions"),
    ("2024-01-23", "AMZN", "Amazon AWS growth rebounds to 17% after cloud slowdown"),
    ("2024-01-24", "AMZN", "Amazon Prime membership crosses 200 million globally"),
    ("2024-01-25", "AMZN", "Amazon faces labor lawsuit over warehouse safety violations"),
    ("2024-01-26", "AMZN", "Amazon logistics division reports 22% efficiency improvement"),
    ("2024-01-29", "AMZN", "Amazon FTC antitrust case expanded to third-party seller fees"),
    ("2024-01-30", "META", "Meta daily active users hit 2.1 billion, up 8% year on year"),
    ("2024-01-31", "META", "Meta AI assistant integrated across all platforms — free access"),
    ("2024-02-01", "META", "Meta faces $1.3B GDPR fine for EU data transfer violations"),
    ("2024-02-02", "META", "Meta Threads gains 130M users in 6 months since launch"),
    ("2024-02-05", "META", "Meta Reality Labs posts $4.6B quarterly loss on VR investment"),
    ("2024-02-06", "NVDA", "Nvidia H100 GPU backlog extends to 12 months on AI demand surge"),
    ("2024-02-07", "NVDA", "Nvidia data center revenue triples year over year to $18.4B"),
    ("2024-02-08", "NVDA", "Nvidia stock hits $600 on AI chip dominance optimism"),
    ("2024-02-09", "NVDA", "China export controls could cut Nvidia revenue by $5B annually"),
    ("2024-02-12", "NVDA", "Nvidia unveils Blackwell GPU architecture for next-gen AI"),
]

df = pd.DataFrame(headlines_data, columns=["date", "ticker", "headline"])
df["date"] = pd.to_datetime(df["date"])
print(f"\n📰 Headlines loaded: {len(df)} articles across {df['ticker'].nunique()} tickers")

# ── 2. TextBlob sentiment ─────────────────────────────────────────
def textblob_sentiment(text):
    blob = TextBlob(text)
    pol  = blob.sentiment.polarity
    sub  = blob.sentiment.subjectivity
    if   pol >  0.1: label = "Positive"
    elif pol < -0.1: label = "Negative"
    else:            label = "Neutral"
    return pd.Series({"tb_polarity": round(pol, 4),
                       "tb_subjectivity": round(sub, 4),
                       "tb_label": label})

df[["tb_polarity","tb_subjectivity","tb_label"]] = (
    df["headline"].apply(textblob_sentiment)
)

# ── 3. VADER sentiment ────────────────────────────────────────────
if VADER_OK:
    sia = SentimentIntensityAnalyzer()
    def vader_sentiment(text):
        scores = sia.polarity_scores(text)
        comp   = scores["compound"]
        if   comp >  0.05: label = "Positive"
        elif comp < -0.05: label = "Negative"
        else:              label = "Neutral"
        return pd.Series({"vader_compound": round(comp, 4),
                           "vader_pos": round(scores["pos"], 4),
                           "vader_neg": round(scores["neg"], 4),
                           "vader_label": label})
    df[["vader_compound","vader_pos","vader_neg","vader_label"]] = (
        df["headline"].apply(vader_sentiment)
    )
    print("✅ VADER sentiment computed")
else:
    df["vader_compound"] = df["tb_polarity"]
    df["vader_label"]    = df["tb_label"]
    print("⚠️  VADER not available — using TextBlob scores")

# ── 4. Composite score & trading signal ───────────────────────────
df["composite_score"] = (df["tb_polarity"] * 0.4 + df["vader_compound"] * 0.6)

def composite_label(score):
    if   score >  0.15: return "Strong Positive"
    elif score >  0.05: return "Positive"
    elif score < -0.15: return "Strong Negative"
    elif score < -0.05: return "Negative"
    return "Neutral"

df["composite_label"] = df["composite_score"].apply(composite_label)

# Trading signal: rolling 3-day average sentiment per ticker
df = df.sort_values(["ticker", "date"])
df["rolling_3d_sentiment"] = (
    df.groupby("ticker")["composite_score"]
    .transform(lambda x: x.rolling(3, min_periods=1).mean())
)

def sentiment_signal(score):
    if   score >  0.10: return "📈 BUY"
    elif score < -0.10: return "📉 SELL"
    return "⏸ HOLD"

df["trading_signal"] = df["rolling_3d_sentiment"].apply(sentiment_signal)

# ── 5. Results summary ────────────────────────────────────────────
print("\n📊 Sentiment Summary by Ticker:")
summary = df.groupby("ticker").agg(
    articles      = ("composite_score","count"),
    avg_sentiment = ("composite_score","mean"),
    positive_pct  = ("composite_label", lambda x: (x.str.contains("Positive")).mean() * 100),
    negative_pct  = ("composite_label", lambda x: (x.str.contains("Negative")).mean() * 100),
).round(3)
print(summary.to_string())

print("\n📰 Top 5 Most Positive Headlines:")
top_pos = df.nlargest(5, "composite_score")[["date","ticker","headline","composite_score"]]
for _, row in top_pos.iterrows():
    print(f"  [{row['ticker']}] {row['headline'][:55]}...  score={row['composite_score']:.3f}")

print("\n📰 Top 5 Most Negative Headlines:")
top_neg = df.nsmallest(5, "composite_score")[["date","ticker","headline","composite_score"]]
for _, row in top_neg.iterrows():
    print(f"  [{row['ticker']}] {row['headline'][:55]}...  score={row['composite_score']:.3f}")

# Latest signals
print("\n📡 Latest Trading Signals (most recent per ticker):")
latest = df.sort_values("date").groupby("ticker").last()[
    ["date","composite_score","rolling_3d_sentiment","trading_signal"]
]
print(latest.to_string())

# ── 6. Five-panel dashboard ───────────────────────────────────────
DARK_BG   = "#1a1d27"
TICKER_COLORS = {"AAPL":"#4f9cf9","MSFT":"#22c55e",
                 "GOOGL":"#f59e0b","AMZN":"#a78bfa","NVDA":"#ef4444"}

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    for sp in ax.spines.values(): sp.set_edgecolor("#2d3142")

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# Panel 1: Sentiment timeline per ticker
ax1 = fig.add_subplot(gs[0, :2])
for ticker, group in df.groupby("ticker"):
    ax1.plot(group["date"], group["composite_score"],
             marker="o", markersize=5, lw=1.5,
             color=TICKER_COLORS.get(ticker, "white"),
             label=ticker, alpha=0.9)
ax1.axhline(0.05,  color="#22c55e", lw=0.8, linestyle="--", alpha=0.6, label="Buy threshold")
ax1.axhline(-0.05, color="#ef4444", lw=0.8, linestyle="--", alpha=0.6, label="Sell threshold")
ax1.axhline(0,     color="white",   lw=0.5, linestyle=":",  alpha=0.4)
ax1.set_ylabel("Composite Sentiment Score")
ax1.legend(fontsize=8, facecolor=DARK_BG, labelcolor="white", ncol=4)
style_ax(ax1, "Composite Sentiment Score Timeline by Ticker", ylabel="Score")

# Panel 2: Average sentiment per ticker
ax2 = fig.add_subplot(gs[0, 2])
avg = df.groupby("ticker")["composite_score"].mean().sort_values()
bar_colors = [TICKER_COLORS.get(t, "gray") for t in avg.index]
bars = ax2.barh(avg.index, avg.values, color=bar_colors,
                edgecolor="#0f1117", linewidth=1.5)
ax2.axvline(0, color="white", lw=0.8, linestyle="--", alpha=0.6)
for bar, v in zip(bars, avg.values):
    ax2.text(v + (0.005 if v >= 0 else -0.005),
             bar.get_y() + bar.get_height()/2,
             f"{v:+.3f}", va="center", color="white", fontsize=9,
             ha="left" if v >= 0 else "right")
ax2.set_xlabel("Average Composite Score")
style_ax(ax2, "Average Sentiment by Ticker")

# Panel 3: Sentiment distribution per ticker
ax3 = fig.add_subplot(gs[1, 0])
label_order = ["Strong Positive","Positive","Neutral","Negative","Strong Negative"]
label_colors = {"Strong Positive":"#16a34a","Positive":"#4ade80",
                "Neutral":"#6b7280","Negative":"#f87171","Strong Negative":"#dc2626"}
dist = df.groupby(["ticker","composite_label"]).size().unstack(fill_value=0)
dist = dist.reindex(columns=[c for c in label_order if c in dist.columns])
dist.plot(kind="bar", ax=ax3, stacked=True,
          color=[label_colors[c] for c in dist.columns],
          edgecolor="#0f1117", linewidth=0.5)
ax3.set_xlabel("Ticker"); ax3.set_ylabel("Article Count")
ax3.tick_params(axis="x", rotation=0)
ax3.legend(fontsize=6, facecolor=DARK_BG, labelcolor="white", loc="upper right")
style_ax(ax3, "Sentiment Label Distribution by Ticker")

# Panel 4: TextBlob vs VADER scatter
ax4 = fig.add_subplot(gs[1, 1])
for ticker, group in df.groupby("ticker"):
    ax4.scatter(group["tb_polarity"], group["vader_compound"],
                color=TICKER_COLORS.get(ticker, "gray"),
                label=ticker, s=55, alpha=0.8, edgecolors="#0f1117", lw=0.5)
ax4.axhline(0, color="white", lw=0.5, linestyle="--", alpha=0.4)
ax4.axvline(0, color="white", lw=0.5, linestyle="--", alpha=0.4)
ax4.set_xlabel("TextBlob Polarity"); ax4.set_ylabel("VADER Compound")
ax4.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")
style_ax(ax4, "TextBlob vs VADER Comparison")

# Panel 5: 3-day rolling sentiment (trading signal overlay)
ax5 = fig.add_subplot(gs[1, 2])
for ticker, group in df.groupby("ticker"):
    ax5.plot(group["date"], group["rolling_3d_sentiment"],
             color=TICKER_COLORS.get(ticker, "white"),
             lw=1.8, label=ticker, alpha=0.9)
ax5.axhline(0.10,  color="#22c55e", lw=0.8, linestyle="--", alpha=0.7, label="BUY 0.10")
ax5.axhline(-0.10, color="#ef4444", lw=0.8, linestyle="--", alpha=0.7, label="SELL -0.10")
ax5.axhline(0, color="white", lw=0.5, linestyle=":", alpha=0.3)
ax5.set_ylabel("3-Day Rolling Sentiment")
ax5.legend(fontsize=6, facecolor=DARK_BG, labelcolor="white", ncol=2)
style_ax(ax5, "3-Day Rolling Sentiment (Signal Generation)")

fig.suptitle("Financial News Sentiment Analysis — NLP Trading Dashboard",
             color="white", fontsize=15, fontweight="bold", y=1.01)
plt.savefig("outputs/04_sentiment_analysis.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("\n📊 Dashboard → outputs/04_sentiment_analysis.png")

print("\n✅ Project 4 — Financial Sentiment Analysis complete.\n")
