"""
============================================================
PROJECT 5: Corporate Bankruptcy Prediction (Altman Z-Score + ML)
============================================================
Industry Relevance (2025-2026):
  - Private Equity firms screen 100s of companies monthly
  - Credit analysts at banks use this for lending decisions
  - Distressed debt funds, rating agencies (Moody's, S&P)
  - Hiring: Credit Risk Analyst | Quant Credit | ML Engineer

Tech Stack:
  Altman Z-Score | scikit-learn | RandomForest | XGBoost |
  GradientBoosting | SHAP-ready | Multi-model comparison |
  Risk Tiering | Actionable Alerts
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("  PROJECT 5: Corporate Bankruptcy Prediction")
print("  Methods: Altman Z-Score + ML Ensemble Models")
print("=" * 65)

# ── 1. Simulate realistic corporate financial dataset ─────────────
np.random.seed(42)
n_healthy, n_bankrupt = 1200, 300   # 20% bankruptcy rate (distressed universe)

# Healthy companies — strong financials
healthy = pd.DataFrame({
    "working_capital_to_assets":  np.random.uniform(0.10, 0.50, n_healthy),
    "retained_earnings_to_assets":np.random.uniform(0.10, 0.60, n_healthy),
    "ebit_to_assets":             np.random.uniform(0.05, 0.25, n_healthy),
    "market_cap_to_liabilities":  np.random.uniform(1.5,  8.0,  n_healthy),
    "revenue_to_assets":          np.random.uniform(0.8,  2.5,  n_healthy),
    "current_ratio":              np.random.uniform(1.5,  4.0,  n_healthy),
    "debt_to_equity":             np.random.uniform(0.1,  1.5,  n_healthy),
    "interest_coverage":          np.random.uniform(3.0,  15.0, n_healthy),
    "net_profit_margin":          np.random.uniform(0.05, 0.30, n_healthy),
    "return_on_assets":           np.random.uniform(0.05, 0.20, n_healthy),
    "bankrupt": 0
})

# Distressed companies — weak financials
bankrupt = pd.DataFrame({
    "working_capital_to_assets":  np.random.uniform(-0.30, 0.08, n_bankrupt),
    "retained_earnings_to_assets":np.random.uniform(-0.50, 0.05, n_bankrupt),
    "ebit_to_assets":             np.random.uniform(-0.15, 0.04, n_bankrupt),
    "market_cap_to_liabilities":  np.random.uniform(0.05,  1.2,  n_bankrupt),
    "revenue_to_assets":          np.random.uniform(0.1,   0.7,  n_bankrupt),
    "current_ratio":              np.random.uniform(0.3,   1.4,  n_bankrupt),
    "debt_to_equity":             np.random.uniform(2.0,   8.0,  n_bankrupt),
    "interest_coverage":          np.random.uniform(0.1,   2.5,  n_bankrupt),
    "net_profit_margin":          np.random.uniform(-0.30, 0.03, n_bankrupt),
    "return_on_assets":           np.random.uniform(-0.20, 0.03, n_bankrupt),
    "bankrupt": 1
})

df = pd.concat([healthy, bankrupt], ignore_index=True).sample(frac=1, random_state=42)
print(f"\n📊 Dataset: {len(df):,} companies")
print(f"   Healthy:   {n_healthy:,} ({n_healthy/len(df)*100:.0f}%)")
print(f"   Bankrupt:  {n_bankrupt:,}  ({n_bankrupt/len(df)*100:.0f}%)")

# ── 2. Altman Z-Score (classic 1968 model) ────────────────────────
# Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
def altman_z_score(row):
    z = (1.2 * row["working_capital_to_assets"] +
         1.4 * row["retained_earnings_to_assets"] +
         3.3 * row["ebit_to_assets"] +
         0.6 * row["market_cap_to_liabilities"] +
         1.0 * row["revenue_to_assets"])
    if   z > 2.99: zone = "Safe"
    elif z > 1.81: zone = "Grey"
    else:          zone = "Distress"
    return pd.Series({"altman_z": round(z, 3), "altman_zone": zone})

df[["altman_z","altman_zone"]] = df.apply(altman_z_score, axis=1)

z_accuracy = (
    ((df["altman_zone"] == "Safe")     & (df["bankrupt"] == 0)) |
    ((df["altman_zone"] == "Distress") & (df["bankrupt"] == 1))
).mean()
print(f"\n📐 Altman Z-Score Classification Accuracy: {z_accuracy*100:.1f}%")
print(f"   Zone distribution:")
print(df["altman_zone"].value_counts().to_string())

# ── 3. ML Models ──────────────────────────────────────────────────
FEATURES = [
    "working_capital_to_assets", "retained_earnings_to_assets",
    "ebit_to_assets", "market_cap_to_liabilities", "revenue_to_assets",
    "current_ratio", "debt_to_equity", "interest_coverage",
    "net_profit_margin", "return_on_assets"
]
X, y = df[FEATURES], df["bankrupt"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, C=1.0
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        max_depth=12, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=250, learning_rate=0.07,
        max_depth=4, random_state=42
    ),
}

results = {}
print("\n🤖 Training ML models...")
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    prob = model.predict_proba(X_test_s)[:, 1]
    auc  = roc_auc_score(y_test, prob)
    ap   = average_precision_score(y_test, prob)
    cv   = cross_val_score(model, X_train_s, y_train,
                           cv=StratifiedKFold(5), scoring="roc_auc").mean()
    results[name] = {
        "model": model, "pred": pred,
        "prob": prob, "auc": auc, "ap": ap, "cv_auc": cv
    }
    print(f"   {name:<25}  AUC={auc:.4f}  AP={ap:.4f}  CV-AUC={cv:.4f}")

best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]
print(f"\n✅ Best model: {best_name}  (AUC={best['auc']:.4f})")
print(f"\n📋 Classification Report ({best_name}):\n")
print(classification_report(y_test, best["pred"],
                             target_names=["Healthy","Bankrupt"]))

# ── 4. Risk tiering ───────────────────────────────────────────────
X_test_df              = X_test.copy().reset_index(drop=True)
X_test_df["prob"]      = best["prob"]
X_test_df["actual"]    = y_test.values
X_test_df["altman_z"]  = df.loc[y_test.index, "altman_z"].values
X_test_df["risk_tier"] = pd.cut(
    X_test_df["prob"],
    bins=[0, 0.20, 0.40, 0.60, 0.80, 1.0],
    labels=["Tier 1 — Safe", "Tier 2 — Low Risk",
            "Tier 3 — Watch", "Tier 4 — High Risk", "Tier 5 — Critical"]
)

print("\n🎯 Bankruptcy Risk Tiering:")
tier = X_test_df.groupby("risk_tier", observed=True).agg(
    count      =("prob","count"),
    avg_prob   =("prob","mean"),
    avg_z_score=("altman_z","mean"),
    actual_bankrupt_rate=("actual","mean")
).round(3)
print(tier.to_string())

# ── 5. Six-panel dashboard ────────────────────────────────────────
DARK_BG = "#1a1d27"
C = {"Logistic Regression":"#4f9cf9","Random Forest":"#22c55e","Gradient Boosting":"#f59e0b"}

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    for sp in ax.spines.values(): sp.set_edgecolor("#2d3142")

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ROC Curves
ax1 = fig.add_subplot(gs[0, 0])
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["prob"])
    ax1.plot(fpr, tpr, color=C[name], lw=2, label=f"{name}\nAUC={r['auc']:.3f}")
ax1.plot([0,1],[0,1],"w--",lw=1,alpha=0.4)
ax1.fill_between(*roc_curve(y_test, best["prob"])[:2], alpha=0.12, color="#22c55e")
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")
style_ax(ax1, "ROC Curves — Model Comparison")

# Feature Importance
ax2 = fig.add_subplot(gs[0, 1])
rf   = results["Random Forest"]["model"]
fi   = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
bar_colors = ["#e74c3c" if v > fi.median() else "#3498db" for v in fi]
ax2.barh(fi.index, fi.values, color=bar_colors)
ax2.set_xlabel("Importance Score")
style_ax(ax2, "Feature Importance (Random Forest)")

# Altman Z-Score distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df[df["bankrupt"]==0]["altman_z"].clip(-2, 8),
         bins=50, alpha=0.75, color="#22c55e", label="Healthy", density=True)
ax3.hist(df[df["bankrupt"]==1]["altman_z"].clip(-2, 8),
         bins=50, alpha=0.75, color="#e74c3c", label="Bankrupt", density=True)
ax3.axvline(1.81, color="orange", lw=1.5, linestyle="--", label="Distress zone (<1.81)")
ax3.axvline(2.99, color="white",  lw=1.5, linestyle="--", label="Safe zone (>2.99)")
ax3.set_xlabel("Altman Z-Score"); ax3.set_ylabel("Density")
ax3.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")
style_ax(ax3, "Altman Z-Score Distribution by Class")

# Confusion Matrix
ax4 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, best["pred"])
im = ax4.imshow(cm, cmap="Blues")
ax4.set_xticks([0,1]); ax4.set_xticklabels(["Healthy","Bankrupt"], color="white", fontsize=9)
ax4.set_yticks([0,1]); ax4.set_yticklabels(["Healthy","Bankrupt"], color="white", fontsize=9)
ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax4.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                 color="white" if cm[i,j]>cm.max()/2 else "black",
                 fontsize=14, fontweight="bold")
style_ax(ax4, f"Confusion Matrix — {best_name}")

# ML Score vs Altman Z
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(X_test_df[X_test_df["actual"]==0]["altman_z"],
            X_test_df[X_test_df["actual"]==0]["prob"],
            color="#22c55e", s=25, alpha=0.6, label="Healthy")
ax5.scatter(X_test_df[X_test_df["actual"]==1]["altman_z"],
            X_test_df[X_test_df["actual"]==1]["prob"],
            color="#e74c3c", s=25, alpha=0.6, label="Bankrupt")
ax5.axvline(1.81, color="orange", lw=1, linestyle="--", alpha=0.7)
ax5.axhline(0.50, color="white",  lw=1, linestyle="--", alpha=0.7)
ax5.set_xlabel("Altman Z-Score"); ax5.set_ylabel("ML Bankruptcy Probability")
ax5.legend(fontsize=8, facecolor=DARK_BG, labelcolor="white")
style_ax(ax5, "ML Probability vs Altman Z-Score")

# Precision-Recall
ax6 = fig.add_subplot(gs[1, 2])
for name, r in results.items():
    prec, rec, _ = precision_recall_curve(y_test, r["prob"])
    ax6.plot(rec, prec, color=C[name], lw=2, label=f"{name} AP={r['ap']:.3f}")
ax6.set_xlabel("Recall"); ax6.set_ylabel("Precision")
ax6.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")
style_ax(ax6, "Precision-Recall Curves")

fig.suptitle("Corporate Bankruptcy Prediction — Altman Z-Score + ML Dashboard",
             color="white", fontsize=15, fontweight="bold", y=1.01)
plt.savefig("outputs/05_bankruptcy_prediction.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("\n📊 Dashboard → outputs/05_bankruptcy_prediction.png")

# ── 6. Live company screening demo ────────────────────────────────
print("\n🔍 Company Screening Demo (5 example companies):")
print("─" * 65)
companies = [
    {"name": "HealthyCorp Inc.",   "vals": [0.35, 0.45, 0.18, 4.2, 1.8, 2.8, 0.4, 10.0, 0.18, 0.14]},
    {"name": "Borderline Ltd.",    "vals": [0.05, 0.08, 0.03, 1.4, 0.6, 1.3, 2.1, 2.0,  0.02, 0.02]},
    {"name": "DistressedCo LLC",   "vals": [-0.20,-0.30,-0.12, 0.3, 0.2, 0.6, 5.5, 0.5, -0.20,-0.15]},
    {"name": "GrowthVentures SA",  "vals": [0.22, 0.12, 0.09, 2.8, 1.2, 2.1, 0.9, 6.0,  0.11, 0.09]},
    {"name": "LegacyCorp plc",     "vals": [-0.05,-0.10, 0.01, 0.8, 0.4, 0.9, 3.8, 1.2, -0.05,-0.03]},
]
rf = results["Random Forest"]["model"]
for co in companies:
    arr   = scaler.transform([co["vals"]])
    prob  = rf.predict_proba(arr)[0][1]
    z_row = {
        "working_capital_to_assets":   co["vals"][0],
        "retained_earnings_to_assets": co["vals"][1],
        "ebit_to_assets":              co["vals"][2],
        "market_cap_to_liabilities":   co["vals"][3],
        "revenue_to_assets":           co["vals"][4],
    }
    z = (1.2*z_row["working_capital_to_assets"] +
         1.4*z_row["retained_earnings_to_assets"] +
         3.3*z_row["ebit_to_assets"] +
         0.6*z_row["market_cap_to_liabilities"] +
         1.0*z_row["revenue_to_assets"])
    risk = ("🔴 CRITICAL"   if prob > 0.70 else
            "🟠 HIGH RISK"  if prob > 0.50 else
            "🟡 WATCH"      if prob > 0.30 else
            "🟢 LOW RISK")
    print(f"  {co['name']:<25}  ML={prob:.3f}  Z={z:.2f}  →  {risk}")

print("\n✅ Project 5 — Bankruptcy Prediction complete.\n")
