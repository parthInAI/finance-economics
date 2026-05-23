<div align="center">

# 💹 Finance & Economics — 5 AI Projects That Actually Matter

**Because money moves the world, and understanding it changes everything**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.nltk.org)
[![yfinance](https://img.shields.io/badge/yfinance-Market%20Data-00C805?style=for-the-badge&logo=yahoo&logoColor=white)](https://pypi.org/project/yfinance)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

</div>

---

## 💡 Why I Built These 5

Finance has always been about trust — trust between people, institutions, and systems. But that trust is fragile. Fraud quietly drains it. Churn silently breaks it. A market crash shatters it overnight. A single word in a news headline can move billions.

I chose these five projects not because they're impressive on paper, but because each one sits at a real fault line in the financial world — a place where a well-built model can quietly protect someone's savings, warn a business before it's too late, or help an ordinary person understand where the market is actually heading.

These aren't toy problems. They're the kinds of questions that keep risk managers up at night and that algorithms are now quietly answering — sometimes better than humans.

---

## 🔍 The Story Behind Each Project

### 01 · Fraud Detection — *Protecting People Who Don't Know They Need Protection*

Most fraud victims don't realise they've been targeted until days later. The damage is done — money gone, trust broken, hours spent on phone calls. A fraud detection model works silently in the background, scoring every transaction in milliseconds, catching the anomaly before it becomes a crisis. This project explores how machines learn to distinguish a genuine late-night purchase from a stolen card being drained across continents. The challenge isn't just accuracy — it's catching the rare, subtle attack without crying wolf on legitimate transactions.

---

### 02 · Customer Churn Prediction — *Listening Before Someone Walks Away*

A customer who is about to leave a bank rarely announces it. The signals are quiet — fewer logins, lower balances, one complaint too many. This project is about learning to listen to those signals before the goodbye. It's not about retention campaigns or growth metrics. It's about genuinely understanding when a relationship is breaking down and giving businesses a chance to respond with something human — a better rate, a personal call, a reason to stay.

---

### 03 · Trading Signal Generation — *Making Sense of the Market's Own Language*

Markets speak in patterns — in the rhythm of moving averages, in the momentum of price swings, in the tension of a Bollinger Band squeeze before a breakout. Traders have read these signals by hand for decades. This project teaches a machine to read that same language — combining MACD, RSI, and Bollinger Bands into a coherent strategy, then testing it honestly against real historical data. Not to get rich, but to understand how structured thinking turns noise into signal.

---

### 04 · Financial Sentiment Analysis — *Words Move Markets*

A CEO's tone in an earnings call, a regulatory headline at 9am, a tweet about supply chain disruption — language shapes financial reality every single day. This project asks a fascinating question: can a machine read the mood of the market from text alone? Using VADER and TextBlob across real headlines from Apple, Microsoft, Google, Amazon, and Nvidia, it explores how sentiment flows through the news cycle and whether it carries a tradable signal. It's NLP meets finance, and the intersection is full of surprises.

---

### 05 · Bankruptcy Prediction — *Seeing the Cliff Before the Edge*

A company rarely collapses overnight. The warning signs are written in the balance sheet — rising debt, shrinking liquidity, interest payments that are starting to hurt. The Altman Z-Score has been predicting bankruptcy since 1968 using just five financial ratios. This project pairs that classic model with modern machine learning to see how much better we can do. The goal isn't morbid — it's protective. Early warning gives stakeholders time to act: restructure, renegotiate, or simply make an informed decision before the fall.

---

## ⚡ Quick Start

```bash
git clone https://github.com/parthInAI/finance-economics.git
cd finance-economics

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download NLTK & TextBlob language data
python -c "import nltk; nltk.download('vader_lexicon')"
python -c "from textblob import download_corpora; download_corpora()"

mkdir -p outputs

# Run any project
python project_01_fraud_detection.py
python project_02_churn_prediction.py
python project_03_trading_signals.py
python project_04_sentiment_analysis.py
python project_05_bankruptcy_prediction.py
```

---

## 🗂️ Project Overview

| # | File | Core Method | Output |
|---|------|-------------|--------|
| 01 | `project_01_fraud_detection.py` | Random Forest · Gradient Boosting · Logistic Regression | ROC, PR-curve, confusion matrix, feature importance |
| 02 | `project_02_churn_prediction.py` | Gradient Boosting · Random Forest · Logistic Regression | Risk bands, retention engine, segment analysis |
| 03 | `project_03_trading_signals.py` | MACD · RSI · Bollinger Bands · Backtesting | 4-panel chart, strategy vs buy-and-hold |
| 04 | `project_04_sentiment_analysis.py` | VADER · TextBlob · Rolling composite score | 5-panel NLP dashboard, BUY/SELL/HOLD signals |
| 05 | `project_05_bankruptcy_prediction.py` | Altman Z-Score · Random Forest · Gradient Boosting | Risk tiering, ML vs Z-Score scatter, screening demo |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| **scikit-learn** | ML models (RF, GB, LR), metrics, preprocessing |
| **pandas / NumPy** | Data manipulation, feature engineering |
| **Matplotlib** | Dark-mode multi-panel dashboards |
| **yfinance** | Real-time & historical stock market data |
| **TextBlob** | NLP — polarity & subjectivity sentiment |
| **NLTK VADER** | Finance-tuned lexicon sentiment scoring |
| **statsmodels** | Time series, econometrics |
| **scipy** | Statistical functions |

---

## 📊 Output Charts

Each project saves a high-resolution dashboard to the `outputs/` folder:

```
outputs/
├── 01_fraud_detection.png        # ROC, PR-curve, confusion matrix, feature importance
├── 02_churn_prediction.png       # Churn by segment, score distribution, model comparison
├── 03_trading_signals.png        # Price + MACD + RSI + portfolio backtest timeline
├── 04_sentiment_analysis.png     # Sentiment timeline, TextBlob vs VADER, signal overlay
└── 05_bankruptcy_prediction.png  # Z-Score distribution, ML vs Altman scatter, risk tiers
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
