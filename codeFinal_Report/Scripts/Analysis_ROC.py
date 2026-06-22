# ==========================================================
# ROC & PR ANALYSIS
# ==========================================================

import matplotlib.pyplot as plt

from sklearn.metrics import (

    roc_curve,

    precision_recall_curve,

    auc
)

from common_ml import *

from xgboost import XGBClassifier

# ==========================================================
# TRAIN MODEL
# ==========================================================

xgb_model = XGBClassifier(

    n_estimators=200,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    scale_pos_weight=10,

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1
)

xgb_model.fit(
    X_train_preprocessed,
    y_train
)

# ==========================================================
# PROBABILITIES
# ==========================================================

y_prob = xgb_model.predict_proba(
    X_test_preprocessed
)[:,1]

# ==========================================================
# ROC CURVE
# ==========================================================

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

# ==========================================================
# PR CURVE
# ==========================================================

precision, recall, _ = precision_recall_curve(
    y_test,
    y_prob
)

pr_auc = auc(
    recall,
    precision
)

# ==========================================================
# ROC PLOT
# ==========================================================

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}"
)

plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# ==========================================================
# PR PLOT
# ==========================================================

plt.figure(figsize=(8,6))

plt.plot(
    recall,
    precision,
    label=f"PR-AUC = {pr_auc:.3f}"
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.legend()

plt.show()

print("\nROC & PR analysis completed!")