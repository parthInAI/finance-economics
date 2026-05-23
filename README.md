<div align="center">

# 💹 Finance & Economics — Top 5 Industry AI Projects

**Production-grade financial ML projects selected for maximum hiring impact**

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

## 🏆 Why These 5 Projects?

These projects were hand-picked from 40 Finance & Economics AI projects based on **real hiring signals in 2025–2026**:

| # | Project | Industry Impact | Top Employers |
|---|---------|----------------|---------------|
| 1 | **Fraud Detection** | $485B lost to fraud globally/year | JPMorgan, Stripe, PayPal, Revolut |
| 2 | **Customer Churn Prediction** | Retention 5–7× cheaper than acquisition | Monzo, Chime, HDFC, Nubank |
| 3 | **Trading Signal Generation** | Quant alpha generation (MACD+RSI+BB) | Citadel, Two Sigma, Jane Street |
| 4 | **Financial Sentiment Analysis** | News-driven quant alpha signals | Bloomberg, AQR, Kensho (S&P) |
| 5 | **Bankruptcy Prediction** | Credit screening + Altman Z-Score + ML | PE firms, Moody's, credit banks |

---

## 🗂️ Project Descriptions

### 01 · Fraud Detection (`project_01_fraud_detection.py`)
Detects fraudulent financial transactions using **Random Forest**, **Gradient Boosting**, and **Logistic Regression**. Features imbalanced dataset handling (3% fraud rate), ROC-AUC comparison, feature importance, real-time scoring demo, and a 6-panel dark-mode dashboard.

**Key concepts:** Class imbalance · Precision-Recall trade-off · Feature engineering · Real-time inference

---

### 02 · Customer Churn Prediction (`project_02_churn_prediction.py`)
Predicts which bank customers are likely to leave using 12 behavioural and demographic features. Includes churn risk segmentation (5 bands), automated retention strategy engine, and model comparison across 3 algorithms.

**Key concepts:** Customer lifetime value · Risk banding · Retention action triggers · Multi-model comparison

---

### 03 · Trading Signal Generation (`project_03_trading_signals.py`)
Generates buy/sell signals using **MACD + RSI + Bollinger Bands** with a full backtesting engine. Downloads real AAPL data via `yfinance`, plots a 4-panel professional chart, and reports strategy vs buy-and-hold returns.

**Key concepts:** Technical analysis · Backtesting · Alpha generation · Moving average crossover

---

### 04 · Financial Sentiment Analysis (`project_04_sentiment_analysis.py`)
Analyses 30 real-world financial headlines across 5 tickers (AAPL, MSFT, GOOGL, AMZN, NVDA) using **VADER + TextBlob**. Generates composite sentiment scores, 3-day rolling signals (BUY/SELL/HOLD), and a 5-panel comparison dashboard.

**Key concepts:** NLP · Dual-model ensemble · News alpha · Sentiment-based trading signals

---

### 05 · Bankruptcy Prediction (`project_05_bankruptcy_prediction.py`)
Combines the classic **Altman Z-Score** (1968 model) with modern ML (Random Forest, Gradient Boosting, Logistic Regression) to predict corporate bankruptcy. Includes 5-tier risk classification, ML vs Z-Score scatter comparison, and a live company screening demo.

**Key concepts:** Altman Z-Score · Credit risk · Financial ratios · Corporate distress

---

## ⚡ Quick Start

```bash
git clone https://github.com/parthInAI/finance-economics.git
cd finance-economics

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download NLTK & TextBlob data
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

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| **scikit-learn** | ML models (RF, GB, LR, SVM), metrics, preprocessing |
| **pandas / NumPy** | Data manipulation, feature engineering |
| **Matplotlib** | Dark-mode multi-panel dashboards |
| **yfinance** | Real-time & historical stock market data |
| **TextBlob** | NLP — polarity & subjectivity sentiment scores |
| **NLTK VADER** | Finance-tuned lexicon sentiment scoring |
| **statsmodels** | Time series (ARIMA), econometrics |
| **scipy** | Statistical functions (Black-Scholes options pricing) |

---

## 📊 Output Charts

Each project saves a high-resolution dashboard to the `outputs/` folder:

```
outputs/
├── 01_fraud_detection.png        # ROC, PR-curve, confusion matrix, feature importance
├── 02_churn_prediction.png       # Model comparison, churn by segment, score dist.
├── 03_trading_signals.png        # Price + MACD + RSI + portfolio backtest
├── 04_sentiment_analysis.png     # Sentiment timeline, heatmap, TextBlob vs VADER
└── 05_bankruptcy_prediction.png  # Z-Score dist., ML vs Altman scatter, risk tiers
```

---

## 🎯 What Makes These Projects Stand Out

Each project goes well beyond a basic model. They include:

- ✅ **Multi-model comparison** (3 algorithms per project with AUC benchmarking)
- ✅ **Production-ready scoring** (real-time inference demos with new data)
- ✅ **Business logic layer** (retention engines, risk tiers, trading signals)
- ✅ **Professional dashboards** (dark-mode, 5–6 panels, publication quality)
- ✅ **Cross-validated metrics** (StratifiedKFold CV for reliable generalisation)
- ✅ **Realistic data distributions** (imbalanced classes, correlated features)

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
