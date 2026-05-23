"""
============================================================
PROJECT 1: Financial Transaction Fraud Detection
============================================================
Industry Relevance (2025-2026):
  - Global fraud losses exceed $485 billion annually
  - Every bank, fintech, and e-commerce company runs fraud ML
  - Top hiring role: ML Engineer - Risk & Fraud (FAANG, JPMorgan,
    Stripe, PayPal, Revolut, Nubank)

Tech Stack:
  scikit-learn | RandomForest | GradientBoosting | SMOTE |
  ROC-AUC | Precision-Recall | Feature Importance | SHAP-ready
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("  PROJECT 1: Financial Transaction Fraud Detection")
print("=" * 65)

# ── 1. Simulate realistic, imbalanced transaction dataset ─────────
np.random.seed(42)
n_legit, n_fraud = 9700, 300   # ~3% fraud rate (realistic)

# Legitimate transactions
legit = pd.DataFrame({
    "transaction_amount":  np.random.lognormal(4.5, 1.0, n_legit),
    "transaction_hour":    np.random.randint(7, 23, n_legit),      # Business hours
    "user_age":            np.random.randint(22, 68, n_legit),
    "account_age_days":    np.random.randint(60, 3000, n_legit),
    "num_txn_last_24h":    np.random.randint(0, 8, n_legit),
    "distance_from_home":  np.random.exponential(20, n_legit),
    "is_international":    np.random.choice([0, 1], n_legit, p=[0.92, 0.08]),
    "failed_pin_attempts": np.random.choice([0, 1, 2], n_legit, p=[0.90, 0.08, 0.02]),
    "is_fraud": 0
})

# Fraudulent transactions — different distributions
fraud = pd.DataFrame({
    "transaction_amount":  np.random.lognormal(6.2, 1.4, n_fraud),   # Higher $ amounts
    "transaction_hour":    np.random.randint(0, 5, n_fraud),          # Odd hours
    "user_age":            np.random.randint(18, 38, n_fraud),
    "account_age_days":    np.random.randint(1, 60, n_fraud),         # New accounts
    "num_txn_last_24h":    np.random.randint(8, 35, n_fraud),         # High velocity
    "distance_from_home":  np.random.exponential(600, n_fraud),
    "is_international":    np.random.choice([0, 1], n_fraud, p=[0.25, 0.75]),
    "failed_pin_attempts": np.random.choice([0, 1, 2], n_fraud, p=[0.40, 0.35, 0.25]),
    "is_fraud": 1
})

df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
print(f"\n📊 Dataset: {len(df):,} transactions")
print(f"   Legitimate: {n_legit:,} ({n_legit/len(df)*100:.1f}%)")
print(f"   Fraudulent: {n_fraud:,}  ({n_fraud/len(df)*100:.1f}%)")

# ── 2. Features & scaling ─────────────────────────────────────────
FEATURES = [
    "transaction_amount", "transaction_hour", "user_age",
    "account_age_days", "num_txn_last_24h",
    "distance_from_home", "is_international", "failed_pin_attempts"
]
X, y = df[FEATURES], df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 3. Train three models ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, C=0.5
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        max_depth=10, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08,
        max_depth=4, random_state=42
    ),
}

results = {}
print("\n🤖 Training models...")
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    prob = model.predict_proba(X_test_s)[:, 1]
    auc  = roc_auc_score(y_test, prob)
    ap   = average_precision_score(y_test, prob)
    results[name] = {
        "model": model, "pred": pred,
        "prob": prob, "auc": auc, "ap": ap
    }
    print(f"   {name:<25}  AUC={auc:.4f}  AP={ap:.4f}")

best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]
print(f"\n✅ Best model: {best_name}")
print(f"\n📋 Classification Report ({best_name}):\n")
print(classification_report(y_test, best["pred"],
                             target_names=["Legitimate", "Fraud"]))

# ── 4. Six-panel dashboard ────────────────────────────────────────
fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
COLORS = {"Logistic Regression": "#4f9cf9", "Random Forest": "#22c55e",
          "Gradient Boosting": "#f59e0b"}
DARK_BG = "#1a1d27"

def style_ax(ax, title):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3142")

# Panel 1: ROC Curves
ax1 = fig.add_subplot(gs[0, 0])
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["prob"])
    ax1.plot(fpr, tpr, color=COLORS[name], lw=2,
             label=f"{name}\nAUC={r['auc']:.3f}")
ax1.plot([0,1],[0,1], "w--", lw=1, alpha=0.4)
ax1.fill_between(*roc_curve(y_test, best["prob"])[:2], alpha=0.15, color="#22c55e")
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")
style_ax(ax1, "ROC Curves — Model Comparison")

# Panel 2: Precision-Recall Curves
ax2 = fig.add_subplot(gs[0, 1])
for name, r in results.items():
    prec, rec, _ = precision_recall_curve(y_test, r["prob"])
    ax2.plot(rec, prec, color=COLORS[name], lw=2,
             label=f"{name} AP={r['ap']:.3f}")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")
style_ax(ax2, "Precision-Recall Curves")

# Panel 3: Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
cm   = confusion_matrix(y_test, best["pred"])
im   = ax3.imshow(cm, cmap="Blues")
tick_labels = ["Legitimate", "Fraud"]
ax3.set_xticks([0,1]); ax3.set_xticklabels(tick_labels, color="white", fontsize=9)
ax3.set_yticks([0,1]); ax3.set_yticklabels(tick_labels, color="white", fontsize=9)
ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax3.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                 color="white" if cm[i,j] > cm.max()/2 else "black",
                 fontsize=14, fontweight="bold")
style_ax(ax3, f"Confusion Matrix — {best_name}")

# Panel 4: Feature Importance
ax4 = fig.add_subplot(gs[1, 0])
rf   = results["Random Forest"]["model"]
fi   = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
bars = ax4.barh(fi.index, fi.values,
                color=["#e74c3c" if v > fi.median() else "#3498db" for v in fi])
ax4.set_xlabel("Importance Score")
style_ax(ax4, "Feature Importance (Random Forest)")

# Panel 5: Fraud Score Distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(best["prob"][y_test == 0], bins=50, alpha=0.75,
         color="#3498db", label="Legitimate", density=True)
ax5.hist(best["prob"][y_test == 1], bins=50, alpha=0.75,
         color="#e74c3c", label="Fraud",      density=True)
ax5.axvline(0.5, color="white", linestyle="--", lw=1.5, label="Threshold 0.5")
ax5.set_xlabel("Fraud Probability Score"); ax5.set_ylabel("Density")
ax5.legend(fontsize=8, facecolor=DARK_BG, labelcolor="white")
style_ax(ax5, "Fraud Score Distribution")

# Panel 6: Transaction Amount by Class
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(df[df["is_fraud"]==0]["transaction_amount"].clip(0, 3000),
         bins=60, alpha=0.75, color="#3498db", label="Legitimate", density=True)
ax6.hist(df[df["is_fraud"]==1]["transaction_amount"].clip(0, 3000),
         bins=60, alpha=0.75, color="#e74c3c", label="Fraud",      density=True)
ax6.set_xlabel("Transaction Amount USD (clipped $3k)")
ax6.legend(fontsize=8, facecolor=DARK_BG, labelcolor="white")
style_ax(ax6, "Amount Distribution by Class")

fig.suptitle("Fraud Detection System — ML Dashboard",
             color="white", fontsize=16, fontweight="bold", y=1.01)
plt.savefig("outputs/01_fraud_detection.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("📊 Dashboard → outputs/01_fraud_detection.png")

# ── 5. Real-time scoring demo ─────────────────────────────────────
print("\n🔍 Real-time Fraud Scoring Demo:")
print("─" * 55)
test_cases = [
    {"desc": "Normal online purchase ($85, 2pm domestic)",
     "vals": [85.0,  14, 35, 900,  2, 12.0,  0, 0]},
    {"desc": "High-value midnight intl. txn ($4,500)",
     "vals": [4500., 2,  24, 15,  18, 850.0, 1, 2]},
    {"desc": "Moderate morning txn ($250)",
     "vals": [250.,  10, 45, 1200, 3, 25.0,  0, 0]},
    {"desc": "New account, rapid txns, far from home",
     "vals": [1800., 3,  21, 8,   20, 720.0, 1, 1]},
]
rf = results["Random Forest"]["model"]
for case in test_cases:
    arr  = scaler.transform([case["vals"]])
    prob = rf.predict_proba(arr)[0][1]
    risk = ("🚨 HIGH RISK"   if prob > 0.6 else
            "⚠️  MEDIUM"     if prob > 0.3 else
            "✅ LOW RISK")
    print(f"  {case['desc'][:48]:<48}  {prob:.3f}  {risk}")

print("\n✅ Project 1 — Fraud Detection complete.\n")
