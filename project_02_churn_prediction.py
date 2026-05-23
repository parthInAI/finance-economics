"""
============================================================
PROJECT 2: Bank Customer Churn Prediction
============================================================
Industry Relevance (2025-2026):
  - Retaining a customer costs 5–7× less than acquiring one
  - Every bank, neobank, BNPL platform runs a churn model
  - Used in: Revolut, Monzo, Chime, JPMorgan, HDFC, SBI
  - Hiring: Data Scientist | CRM Analytics | ML Engineer

Tech Stack:
  scikit-learn | GradientBoosting | RandomForest | ROC-AUC |
  Retention Strategy Engine | Segment Analysis | Visualization
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("  PROJECT 2: Bank Customer Churn Prediction")
print("=" * 65)

# ── 1. Realistic customer dataset ────────────────────────────────
np.random.seed(42)
n = 10_000

df = pd.DataFrame({
    "age":               np.random.randint(18, 75, n),
    "account_balance":   np.abs(np.random.normal(15000, 10000, n)),
    "num_products":      np.random.choice([1,2,3,4], n, p=[0.50, 0.32, 0.14, 0.04]),
    "credit_score":      np.random.randint(300, 850, n),
    "tenure_years":      np.random.randint(0, 20, n),
    "monthly_txn":       np.random.poisson(14, n),
    "has_credit_card":   np.random.choice([0,1], n, p=[0.28, 0.72]),
    "is_active_member":  np.random.choice([0,1], n, p=[0.42, 0.58]),
    "estimated_salary":  np.abs(np.random.normal(70000, 28000, n)),
    "num_complaints":    np.random.choice([0,1,2,3], n, p=[0.70, 0.20, 0.07, 0.03]),
    "geography":         np.random.choice(["France","Germany","Spain","UK"],
                                          n, p=[0.40, 0.22, 0.20, 0.18]),
    "gender":            np.random.choice(["Male","Female"], n),
})

# Churn probability = realistic weighted combination
churn_prob = (
    0.12 * (df["age"] > 55).astype(int) +
    0.18 * (df["num_products"] == 1).astype(int) +
    0.14 * (df["is_active_member"] == 0).astype(int) +
    0.10 * (df["credit_score"] < 420).astype(int) +
    0.08 * (df["tenure_years"] < 2).astype(int) +
    0.06 * (df["geography"] == "Germany").astype(int) +
    0.12 * (df["num_complaints"] >= 2).astype(int) +
    np.random.uniform(0, 0.18, n)
)
df["churned"] = (churn_prob > 0.30).astype(int)

print(f"\n📊 Dataset: {len(df):,} customers")
print(f"   Churn rate: {df['churned'].mean()*100:.1f}%")

# ── 2. Preprocessing ──────────────────────────────────────────────
for col in ["geography", "gender"]:
    df[col + "_enc"] = LabelEncoder().fit_transform(df[col])

FEATURES = [
    "age", "account_balance", "num_products", "credit_score",
    "tenure_years", "monthly_txn", "has_credit_card",
    "is_active_member", "estimated_salary", "num_complaints",
    "geography_enc", "gender_enc"
]
X, y = df[FEATURES], df["churned"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler   = StandardScaler()
Xtr_s    = scaler.fit_transform(X_train)
Xte_s    = scaler.transform(X_test)

# ── 3. Train models ───────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        max_depth=12, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=250, learning_rate=0.06,
        max_depth=4, random_state=42
    ),
}
results = {}
print("\n🤖 Training models...")
for name, m in models.items():
    m.fit(Xtr_s, y_train)
    prob = m.predict_proba(Xte_s)[:, 1]
    pred = m.predict(Xte_s)
    auc  = roc_auc_score(y_test, prob)
    results[name] = {"model": m, "prob": prob, "pred": pred, "auc": auc}
    print(f"   {name:<25}  AUC = {auc:.4f}")

best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]
print(f"\n✅ Best: {best_name}  (AUC={best['auc']:.4f})")
print(f"\n📋 Classification Report:\n")
print(classification_report(y_test, best["pred"],
                             target_names=["Retained", "Churned"]))

# ── 4. Risk segmentation ─────────────────────────────────────────
X_test_df               = X_test.copy().reset_index(drop=True)
X_test_df["churn_prob"] = best["prob"]
X_test_df["actual"]     = y_test.values
X_test_df["risk_band"]  = pd.cut(
    X_test_df["churn_prob"],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=["Very Low","Low","Medium","High","Very High"]
)
print("\n🎯 Churn Risk Segmentation:")
seg = X_test_df.groupby("risk_band", observed=True).agg(
    customers    =("churn_prob","count"),
    avg_prob     =("churn_prob","mean"),
    actual_churn =("actual","mean")
).round(3)
print(seg.to_string())

# ── 5. Six-panel dashboard ────────────────────────────────────────
DARK_BG = "#1a1d27"
C = {"Logistic Regression":"#4f9cf9","Random Forest":"#22c55e","Gradient Boosting":"#f59e0b"}

def style_ax(ax, title):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for sp in ax.spines.values(): sp.set_edgecolor("#2d3142")

fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ROC
ax = fig.add_subplot(gs[0, 0])
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["prob"])
    ax.plot(fpr, tpr, color=C[name], lw=2, label=f"{name} ({r['auc']:.3f})")
ax.plot([0,1],[0,1],"w--",lw=1,alpha=0.4)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")
style_ax(ax, "ROC Curves")

# Feature importance
ax = fig.add_subplot(gs[0, 1])
fi = pd.Series(results["Random Forest"]["model"].feature_importances_,
               index=FEATURES).sort_values()
colors_fi = ["#e74c3c" if v > fi.median() else "#3498db" for v in fi]
ax.barh(fi.index, fi.values, color=colors_fi)
ax.set_xlabel("Importance")
style_ax(ax, "Feature Importance (Random Forest)")

# Churn score dist
ax = fig.add_subplot(gs[0, 2])
ax.hist(best["prob"][y_test==0], bins=50, alpha=0.75, color="#2ecc71",
        label="Retained", density=True)
ax.hist(best["prob"][y_test==1], bins=50, alpha=0.75, color="#e74c3c",
        label="Churned",  density=True)
ax.axvline(0.5, color="white", linestyle="--", lw=1.5, label="Threshold")
ax.set_xlabel("Churn Probability"); ax.set_ylabel("Density")
ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor="white")
style_ax(ax, "Churn Score Distribution")

# Confusion matrix
ax = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, best["pred"])
im = ax.imshow(cm, cmap="Oranges")
ax.set_xticks([0,1]); ax.set_xticklabels(["Retained","Churned"], color="white", fontsize=9)
ax.set_yticks([0,1]); ax.set_yticklabels(["Retained","Churned"], color="white", fontsize=9)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                color="white" if cm[i,j]>cm.max()/2 else "black",
                fontsize=14, fontweight="bold")
style_ax(ax, f"Confusion Matrix — {best_name}")

# Churn by num_products
ax = fig.add_subplot(gs[1, 1])
cr = df.groupby("num_products")["churned"].mean() * 100
bar_colors = ["#3498db","#e67e22","#e74c3c","#9b59b6"]
bars = ax.bar(cr.index, cr.values, color=bar_colors, edgecolor="#0f1117", linewidth=1.5)
ax.set_xlabel("Number of Products"); ax.set_ylabel("Churn Rate (%)")
ax.tick_params(axis="x", rotation=0)
for bar, v in zip(bars, cr.values):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.3,
            f"{v:.1f}%", ha="center", color="white", fontsize=9, fontweight="bold")
style_ax(ax, "Churn Rate by Number of Products")

# Churn by age group
ax = fig.add_subplot(gs[1, 2])
df["age_group"] = pd.cut(df["age"], bins=[18,30,40,50,60,75],
                          labels=["18-30","31-40","41-50","51-60","61-75"])
cr_age = df.groupby("age_group", observed=True)["churned"].mean() * 100
ax.bar(range(len(cr_age)), cr_age.values, color="#1abc9c",
       edgecolor="#0f1117", linewidth=1.5)
ax.set_xticks(range(len(cr_age)))
ax.set_xticklabels(cr_age.index, rotation=30, color="white", fontsize=8)
ax.set_ylabel("Churn Rate (%)")
style_ax(ax, "Churn Rate by Age Group")

fig.suptitle("Customer Churn Prediction — ML Dashboard",
             color="white", fontsize=16, fontweight="bold", y=1.01)
plt.savefig("outputs/02_churn_prediction.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("\n📊 Dashboard → outputs/02_churn_prediction.png")

# ── 6. Automated retention strategy engine ────────────────────────
print("\n💡 Automated Retention Strategy Engine:")
print("─" * 55)
X_test_df["age"]           = X_test["age"].values
X_test_df["num_products"]  = X_test["num_products"].values
X_test_df["is_active_member"] = X_test["is_active_member"].values
X_test_df["account_balance"]  = X_test["account_balance"].values
X_test_df["num_complaints"]   = X_test["num_complaints"].values

high_risk = X_test_df[X_test_df["churn_prob"] > 0.65].nlargest(5, "churn_prob")
for _, row in high_risk.iterrows():
    strategies = []
    if row["num_products"] == 1:
        strategies.append("→ Cross-sell: savings account or insurance product")
    if row["is_active_member"] == 0:
        strategies.append("→ Re-engagement: personalised email/push campaign")
    if row["account_balance"] < 5000:
        strategies.append("→ Offer: premium interest rate to grow balance")
    if row["num_complaints"] >= 2:
        strategies.append("→ Escalate: assign dedicated relationship manager")
    if not strategies:
        strategies.append("→ Loyalty reward: cashback or fee waiver")
    print(f"\n  Customer (age={int(row['age'])}) "
          f"Churn Risk: {row['churn_prob']:.1%}")
    for s in strategies:
        print(f"    {s}")

print("\n✅ Project 2 — Customer Churn Prediction complete.\n")
