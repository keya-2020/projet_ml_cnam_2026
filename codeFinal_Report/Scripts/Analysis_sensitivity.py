# ==========================================================
# SENSITIVITY ANALYSIS
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.metrics import (

    precision_score,

    recall_score,

    f1_score,

    fbeta_score
)

from xgboost import XGBClassifier

from common_ml import *

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
# THRESHOLDS
# ==========================================================

thresholds = np.arange(

    0.05,

    0.55,

    0.05
)

results = []

# ==========================================================
# LOOP
# ==========================================================

for threshold in thresholds:

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    f2 = fbeta_score(
        y_test,
        y_pred,
        beta=2
    )

    results.append({

        "threshold": threshold,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "f2_score": f2
    })

# ==========================================================
# RESULTS DF
# ==========================================================

results_df = pd.DataFrame(
    results
)

print(results_df)

# ==========================================================
# SAVE
# ==========================================================

results_df.to_csv(

    "threshold_sensitivity.csv",

    index=False
)

print("\nSensitivity analysis completed!")