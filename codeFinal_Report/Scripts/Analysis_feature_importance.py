# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from common_ml import *

# ==========================================================
# TRAIN MODEL
# ==========================================================

print("===================================================")
print("FEATURE IMPORTANCE ANALYSIS")
print("===================================================")

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
# FEATURE IMPORTANCE
# ==========================================================

importance_df = pd.DataFrame({

    "feature": X.columns,

    "importance": xgb_model.feature_importances_
})

importance_df = importance_df.sort_values(

    by="importance",

    ascending=False
)

print("\nTop Features:")

print(importance_df.head(15))

# ==========================================================
# SAVE CSV
# ==========================================================

importance_df.to_csv(
    "xgb_feature_importance.csv",
    index=False
)

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(10,6))

plt.barh(

    importance_df["feature"][:15],

    importance_df["importance"][:15]
)

plt.gca().invert_yaxis()

plt.title("XGBoost Feature Importance")

plt.xlabel("Importance")

plt.tight_layout()

plt.show()

print("\nFeature importance completed!")