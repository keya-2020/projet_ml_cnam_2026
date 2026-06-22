# ==========================================================
# SHAP ANALYSIS WITH NATIVE XGBOOST
# ==========================================================

# ==========================================================
# IMPORTS
# ==========================================================

import xgboost as xgb
import shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from common_ml import *

# ==========================================================
# TRAIN MODEL
# ==========================================================

print("===================================================")
print("SHAP ANALYSIS")
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

print("\nTraining XGBoost...")

xgb_model.fit(
    X_train_preprocessed,
    y_train
)

print("Training completed!")

# ==========================================================
# SAMPLE
# ==========================================================

sample_size = 2000

X_sample = X_test_preprocessed[:sample_size]

feature_names = X.columns.tolist()

# ==========================================================
# CONVERT TO DMATRIX
# ==========================================================

dmatrix = xgb.DMatrix(

    X_sample,

    feature_names=feature_names
)

# ==========================================================
# GET BOOSTER
# ==========================================================

booster = xgb_model.get_booster()

# ==========================================================
# NATIVE SHAP VALUES
# ==========================================================

print("\nComputing native XGBoost SHAP values...")

shap_values = booster.predict(

    dmatrix,

    pred_contribs=True
)

# ==========================================================
# REMOVE BIAS COLUMN
# ==========================================================

shap_values = shap_values[:, :-1]

print("SHAP values computed!")

# ==========================================================
# SUMMARY PLOT
# ==========================================================

print("\nGenerating SHAP summary plot...")

shap.summary_plot(

    shap_values,

    X_sample,

    feature_names=feature_names
)

# ==========================================================
# BAR PLOT
# ==========================================================

print("\nGenerating SHAP bar plot...")

shap.summary_plot(

    shap_values,

    X_sample,

    feature_names=feature_names,

    plot_type="bar"
)

# ==========================================================
# FEATURE IMPORTANCE DF
# ==========================================================

mean_abs_shap = np.abs(
    shap_values
).mean(axis=0)

importance_df = pd.DataFrame({

    "feature": feature_names,

    "mean_abs_shap": mean_abs_shap
})

importance_df = importance_df.sort_values(

    by="mean_abs_shap",

    ascending=False
)

# ==========================================================
# DISPLAY TOP FEATURES
# ==========================================================

print("\n===================================================")
print("TOP SHAP FEATURES")
print("===================================================")

print(
    importance_df.head(15)
)

# ==========================================================
# SAVE CSV
# ==========================================================

importance_df.to_csv(

    "shap_feature_importance.csv",

    index=False
)

print("\nSHAP feature importance saved!")

# ==========================================================
# DEPENDENCE PLOT
# ==========================================================

top_feature = importance_df.iloc[0]["feature"]

print(f"\nGenerating dependence plot for: {top_feature}")

shap.dependence_plot(

    top_feature,

    shap_values,

    X_sample,

    feature_names=feature_names
)

# ==========================================================
# FINAL
# ==========================================================

print("\n===================================================")
print("SHAP ANALYSIS COMPLETED")
print("===================================================")