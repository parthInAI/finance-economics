# Finance and Economics AI - 5 Projects

Five end-to-end quantitative models applied to real financial problems. Each project follows a full ML pipeline: data acquisition, EDA, feature engineering, model training, evaluation, and business-level interpretation of results.

---

## Projects

### 1. Fraud Detection on Imbalanced Data

Detects fraudulent transactions from a heavily imbalanced dataset (fraud cases are less than 1% of records).

- SMOTE oversampling and class weighting strategies
- XGBoost and Random Forest with threshold tuning
- Precision-recall optimisation over accuracy (accuracy is misleading on imbalanced data)
- Feature importance analysis identifying top fraud signals

**Business relevance:** False negatives cost money; false positives cost customers. This project models the real trade-off.

---

### 2. Customer Churn Prediction with Retention Engine

Predicts which customers are likely to leave and automatically generates retention recommendations.

- Logistic regression, gradient boosting, and neural network comparison
- SHAP values for per-customer explainability
- Automated retention action engine based on churn probability and customer segment
- Cohort analysis identifying highest-risk customer groups

**Business relevance:** Retaining a customer costs a fraction of acquiring a new one. Explainability matters because retention actions differ by churn reason.

---

### 3. Algorithmic Trading with Backtesting

Generates buy/sell signals using technical indicators and backtests on real historical price data.

- MACD and RSI signal generation
- yfinance API for real market data (5 tickers)
- Backtesting engine with transaction cost modelling
- Sharpe ratio, drawdown, and win rate performance metrics

**Business relevance:** A strategy that looks good in-sample often fails out-of-sample. The backtesting framework here enforces walk-forward validation.

---

### 4. Financial News Sentiment Analysis

Classifies financial news headlines as positive, negative, or neutral and correlates sentiment with price movement.

- VADER sentiment scoring on 5 tickers
- NLTK preprocessing pipeline
- Sentiment-price correlation analysis with lag testing
- Rolling sentiment window for trend detection

**Business relevance:** Market sentiment is a leading indicator. This project quantifies how news sentiment correlates with short-term price movement.

---

### 5. Corporate Bankruptcy Prediction

Predicts corporate bankruptcy risk by combining classical financial ratios with machine learning.

- Altman Z-Score as a feature (classical bankruptcy model from 1968)
- Gradient boosting trained on financial statement data
- Comparison of classical vs ML approaches
- Risk scoring output for portfolio screening

**Business relevance:** The Altman Z-Score is still widely used in credit analysis. This project shows when ML meaningfully outperforms the classical model and when it does not.

---

## Tech stack

| Component | Technology |
|---|---|
| Modelling | scikit-learn, XGBoost, gradient boosting |
| Market data | yfinance |
| NLP | NLTK, VADER |
| Classical models | Altman Z-Score |
| Explainability | SHAP |
| Data processing | pandas, NumPy |

---

## Getting started

```bash
git clone https://github.com/parthInAI/finance-economics
cd finance-economics

pip install -r requirements.txt

# Each project has its own notebook
jupyter notebook fraud_detection/fraud_detection.ipynb
jupyter notebook churn/churn_prediction.ipynb
jupyter notebook trading/algorithmic_trading.ipynb
jupyter notebook sentiment/sentiment_analysis.ipynb
jupyter notebook bankruptcy/bankruptcy_prediction.ipynb
```

---

## Skills demonstrated

- Imbalanced classification and threshold optimisation
- SHAP explainability for business decisions
- Technical indicator engineering and backtesting
- Sentiment analysis on financial text
- Classical and ML model comparison
- End-to-end pipeline from raw data to business insight

---

## Related projects

- [Term Deposit Prediction](https://github.com/parthInAI/Data-Analytics-term-deposit-prediction) — banking analytics
- [Portfolio](https://parthinai.github.io/)
