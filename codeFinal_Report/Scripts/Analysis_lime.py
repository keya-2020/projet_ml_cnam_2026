# ==========================================================
# LIME ANALYSIS
# ==========================================================

import lime
import lime.lime_tabular

import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from common_ml import *

# ==========================================================
# TRAIN MODEL
# ==========================================================

print("===================================================")
print("LIME ANALYSIS")
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
# LIME EXPLAINER
# ==========================================================

explainer = lime.lime_tabular.LimeTabularExplainer(

    training_data=np.array(
        X_train_preprocessed
    ),

    feature_names=X.columns.tolist(),

    class_names=["Normal", "Fraud"],

    mode="classification"
)

# ==========================================================
# SELECT INSTANCE
# ==========================================================

instance_id = 10

instance = X_test_preprocessed[
    instance_id
]

# ==========================================================
# EXPLANATION
# ==========================================================

explanation = explainer.explain_instance(

    instance,

    xgb_model.predict_proba,

    num_features=10
)

# ==========================================================
# DISPLAY
# ==========================================================

print("\nLIME Explanation:")

for feature, weight in explanation.as_list():

    print(f"{feature}: {weight:.4f}")

# ==========================================================
# SAVE HTML
# ==========================================================

explanation.save_to_file(
    "lime_explanation.html"
)

print("\nLIME explanation saved!")