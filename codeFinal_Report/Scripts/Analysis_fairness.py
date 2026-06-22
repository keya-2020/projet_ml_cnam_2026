# ==========================================================
# FAIRNESS METRICS
# ==========================================================

import pandas as pd

from sklearn.metrics import (
    recall_score
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
# PREDICTIONS
# ==========================================================

y_pred = xgb_model.predict(
    X_test_preprocessed
)

# ==========================================================
# TEST DF
# ==========================================================

test_df = X_test.copy()

test_df["fraud_bool"] = y_test.values

test_df["prediction"] = y_pred

# ==========================================================
# GROUP FAIRNESS
# ==========================================================

group_col = "housing_status"

print("===================================================")
print("FAIRNESS ANALYSIS")
print("===================================================")

groups = test_df[group_col].unique()

fairness_results = []

for g in groups:

    subset = test_df[
        test_df[group_col] == g
    ]

    recall = recall_score(

        subset["fraud_bool"],

        subset["prediction"],

        zero_division=0
    )

    fairness_results.append({

        "group": g,

        "recall": recall,

        "count": len(subset)
    })

fairness_df = pd.DataFrame(
    fairness_results
)

print(fairness_df)

fairness_df.to_csv(

    "fairness_results.csv",

    index=False
)

print("\nFairness analysis completed!")