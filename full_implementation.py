"""
==============================================================================
 SHAP-BASED INTERPRETABLE ML FOR UNIVERSITY FOOTBALL INJURY RISK PREDICTION
 Case-study reproduction of: Ma, J., Liu, S. & Pei, Y. (2025).
 "SHAP-based interpretable machine learning for injury risk prediction in
 university football players." Scientific Reports, 15, 40252.
==============================================================================

NOTE ON THE DATA
-----------------
The authors' raw Kaggle CSV was not distributed with the paper. To still
demonstrate the complete pipeline end-to-end, this script REGENERATES a
statistically faithful surrogate dataset: every one of the 18 predictors is
sampled from Normal(mean, sd) using the exact per-group (injured / not
injured) means and standard deviations published in Table 2 of the paper.
Injury labels then come from the two randomly assigned 400-athlete groups,
so the surrogate reproduces the *marginal* statistics and the resulting
group separability, without being the authors' original 800 rows.
This is disclosed explicitly and is standard practice when a published
dataset is not attached / access-restricted but its descriptive statistics
are reported in full.

PIPELINE STAGES
---------------
1. Synthetic data generation matching Table 2 statistics
2. Preprocessing (one-hot encoding of Position, Z-score standardisation)
3. Stratified 70/30 train-test split
4. Training + comparison of 10 ML algorithms (same list as the paper)
5. Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier score
6. SHAP (KernelExplainer) interpretability analysis of the best model
7. Visualisations: performance heatmap, ROC curves, confusion matrix,
   SHAP bar / beeswarm plots, feature-correlation heatmap
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score, confusion_matrix,
                              roc_curve, brier_score_loss)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# ----------------------------------------------------------------------------
# STAGE 1 — Synthetic data generation (matches Table 2 of the paper)
# ----------------------------------------------------------------------------
N_PER_GROUP = 400
# feature: (mean_noninjured, sd_noninjured, mean_injured, sd_injured)
stats = {
    "Age":                        (20.5, 1.8, 20.6, 1.9),
    "Height_cm":                  (176.2, 6.8, 175.8, 7.1),
    "Weight_kg":                  (71.3, 8.5, 71.7, 8.9),
    "BMI":                        (22.9, 2.1, 23.1, 2.3),
    "Training_Hours_Per_Week":    (11.2, 3.4, 11.8, 3.6),
    "Matches_Played_Past_Season": (18.5, 6.2, 19.3, 6.5),
    "Previous_Injury_Count":      (0.8, 1.1, 1.4, 1.3),
    "Knee_Strength_Score":        (78.3, 12.5, 74.2, 13.8),
    "Hamstring_Flexibility":      (18.5, 5.2, 16.8, 5.8),
    "Reaction_Time_ms":           (285, 45, 305, 52),
    "Balance_Test_Score":         (82.4, 14.2, 71.8, 16.5),
    "Sprint_Speed_10m_s":         (1.82, 0.15, 1.89, 0.17),
    "Agility_Score":              (75.8, 11.3, 71.2, 12.8),
    "Sleep_Hours_Per_Night":      (7.8, 1.1, 6.9, 1.3),
    "Stress_Level_Score":         (15.2, 5.8, 22.8, 6.5),
    "Nutrition_Quality_Score":    (72.5, 12.8, 68.3, 14.2),
    "Warmup_Routine_Adherence":   (88.5, 11.2, 83.2, 13.5),
}
positions = ["Forward", "Midfielder", "Defender", "Goalkeeper"]

def sample_group(n, which):
    d = {}
    for feat, (m0, s0, m1, s1) in stats.items():
        m, s = (m0, s0) if which == 0 else (m1, s1)
        d[feat] = rng.normal(m, s, n)
    d["Position"] = rng.choice(positions, n, p=[0.3, 0.3, 0.3, 0.1])
    d["Injury_Next_Season"] = which
    return pd.DataFrame(d)

data = pd.concat([sample_group(N_PER_GROUP, 0), sample_group(N_PER_GROUP, 1)],
                  ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)
data["Age"] = data["Age"].clip(18, 24).round(0)
data["Previous_Injury_Count"] = data["Previous_Injury_Count"].clip(lower=0).round(0)
data["Sleep_Hours_Per_Night"] = data["Sleep_Hours_Per_Night"].clip(4, 10)
data["Warmup_Routine_Adherence"] = data["Warmup_Routine_Adherence"].clip(0, 100)

# ----------------------------------------------------------------------------
# STAGE 2 — Preprocessing
# ----------------------------------------------------------------------------
df = pd.get_dummies(data, columns=["Position"], prefix="Position")
y = df["Injury_Next_Season"].values
X = df.drop(columns=["Injury_Next_Season"])
feature_names = X.columns.tolist()

# ----------------------------------------------------------------------------
# STAGE 3 — Train / test split + standardisation
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

# ----------------------------------------------------------------------------
# STAGE 4 — Train & compare 10 algorithms
# ----------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0),
    "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
}

results, roc_data, fitted = [], {}, {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    fitted[name] = model
    pred = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)[:, 1]
    results.append(dict(
        Model=name,
        Accuracy=accuracy_score(y_test, pred),
        Precision=precision_score(y_test, pred),
        Recall=recall_score(y_test, pred),
        F1=f1_score(y_test, pred),
        ROC_AUC=roc_auc_score(y_test, proba),
        PR_AUC=average_precision_score(y_test, proba),
        Brier=brier_score_loss(y_test, proba),
    ))
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_data[name] = (fpr, tpr)

res_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
best_name = res_df.iloc[0]["Model"]
best_model = fitted[best_name]
print(res_df.round(3).to_string(index=False))
print("\nBest model:", best_name)

# ----------------------------------------------------------------------------
# STAGE 5 — SHAP interpretability analysis of the best model
# ----------------------------------------------------------------------------
background = shap.sample(X_train_s, 60, random_state=RANDOM_STATE)
explain_set = X_test_s.sample(100, random_state=RANDOM_STATE)
explainer = shap.KernelExplainer(lambda x: best_model.predict_proba(x)[:, 1], background)
shap_values = explainer.shap_values(explain_set, nsamples=100)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_order = np.argsort(mean_abs_shap)[::-1]
top_features = pd.DataFrame({
    "Feature": np.array(feature_names)[importance_order],
    "MeanAbsSHAP": mean_abs_shap[importance_order],
}).head(10)
print("\nTop-10 SHAP feature importances:\n", top_features.to_string(index=False))

# ----------------------------------------------------------------------------
# STAGE 6 — Visualisations
# ----------------------------------------------------------------------------
# (a) Performance heatmap
heat = res_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]]
plt.figure(figsize=(7, 5))
plt.imshow(heat.values, cmap="Blues", vmin=0.6, vmax=1.0, aspect="auto")
plt.xticks(range(len(heat.columns)), heat.columns, rotation=30, ha="right")
plt.yticks(range(len(heat.index)), heat.index)
plt.title("Model Performance Heatmap"); plt.colorbar(fraction=0.03)
plt.tight_layout(); plt.savefig("performance_heatmap.png", dpi=160)

# (b) ROC curves
plt.figure(figsize=(6.5, 5.5))
for name, (fpr, tpr) in roc_data.items():
    auc = res_df.loc[res_df.Model == name, "ROC_AUC"].values[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison"); plt.legend(fontsize=7)
plt.tight_layout(); plt.savefig("roc_curves.png", dpi=160)

# (c) Confusion matrix (best model)
cm = confusion_matrix(y_test, best_model.predict(X_test_s))
plt.figure(figsize=(4, 3.6))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xticks([0, 1], ["Low risk", "High risk"]); plt.yticks([0, 1], ["Low risk", "High risk"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix – {best_name}")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=160)

# (d) SHAP bar + beeswarm
plt.figure(figsize=(7, 7))
plt.barh(top_features.Feature[::-1], top_features.MeanAbsSHAP[::-1])
plt.xlabel("mean(|SHAP value|)"); plt.title(f"SHAP Feature Importance – {best_name}")
plt.tight_layout(); plt.savefig("shap_bar.png", dpi=160)

shap.summary_plot(shap_values, explain_set, feature_names=feature_names, show=False)
plt.tight_layout(); plt.savefig("shap_beeswarm.png", dpi=160, bbox_inches="tight")

print("\nAll figures saved to current directory.")
