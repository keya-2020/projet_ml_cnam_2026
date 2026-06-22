# ==========================================================
# STACKING ENSEMBLE
# Logistic + XGBoost + LightGBM
# ==========================================================

import time

from sklearn.ensemble import (
    StackingClassifier
)

from sklearn.linear_model import (
    LogisticRegression
)

from xgboost import (
    XGBClassifier
)

from lightgbm import (
    LGBMClassifier
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from common_ml import *

# ==========================================================
# START TIMER
# ==========================================================

start_time = time.time()

print("===================================================")
print("STACKING ENSEMBLE TRAINING")
print("===================================================")

# ==========================================================
# BASE MODELS
# ==========================================================

base_models = [

    (
        "logistic",

        LogisticRegression(

            class_weight="balanced",

            max_iter=1000,

            random_state=42,

            n_jobs=-1
        )
    ),

    (
        "xgb",

        XGBClassifier(

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
    ),

    (
        "lgbm",

        LGBMClassifier(

            n_estimators=200,

            max_depth=6,

            learning_rate=0.05,

            subsample=0.8,

            colsample_bytree=0.8,

            scale_pos_weight=10,

            objective="binary",

            random_state=42,

            n_jobs=-1,

            verbose=-1
        )
    )
]

# ==========================================================
# META MODEL
# ==========================================================

meta_model = LogisticRegression(

    class_weight="balanced",

    max_iter=1000,

    random_state=42
)

# ==========================================================
# STACKING MODEL
# ==========================================================

stacking_model = StackingClassifier(

    estimators=base_models,

    final_estimator=meta_model,

    stack_method="predict_proba",

    passthrough=True,

    cv=5,

    n_jobs=-1
)

# ==========================================================
# TRAINING
# ==========================================================

stacking_model.fit(

    X_train_preprocessed,

    y_train
)

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred_stack = stacking_model.predict(

    X_test_preprocessed
)

y_prob_stack = stacking_model.predict_proba(

    X_test_preprocessed

)[:,1]

# ==========================================================
# END TIMER
# ==========================================================

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("\n===================================================")
print("STACKING RESULTS")
print("===================================================")

print("\nConfusion Matrix:")

print(

    confusion_matrix(

        y_test,

        y_pred_stack
    )
)

print("\nClassification Report:")

print(

    classification_report(

        y_test,

        y_pred_stack
    )
)

print(

    "\nROC-AUC:",

    roc_auc_score(

        y_test,

        y_prob_stack
    )
)

print(

    "\nComputation Time:",

    round(
        end_time - start_time,
        2
    ),

    "seconds"
)

# ==========================================================
# SAVE
# ==========================================================

save_results(

    model_name="Stacking XGB+LGBM+LR",

    y_pred=y_pred_stack,

    y_prob=y_prob_stack,

    start_time=start_time,

    end_time=end_time
)

print("\nStacking Ensemble completed!")