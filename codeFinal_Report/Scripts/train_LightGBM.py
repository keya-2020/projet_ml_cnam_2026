# ==========================================================
# LIGHTGBM
# ==========================================================

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import time

from common_ml import *

# ==========================================================
# TRAIN MODEL
# ==========================================================

start_time = time.time()

lgbm_model = LGBMClassifier(

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

lgbm_model.fit(
    X_train_preprocessed,
    y_train
)

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred_lgbm = lgbm_model.predict(
    X_test_preprocessed
)

y_prob_lgbm = lgbm_model.predict_proba(
    X_test_preprocessed
)[:, 1]

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("LightGBM Results")

print(confusion_matrix(
    y_test,
    y_pred_lgbm
))

print(classification_report(
    y_test,
    y_pred_lgbm
))

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        y_prob_lgbm
    )
)

# ==========================================================
# SAVE
# ==========================================================

save_results(
    model_name="LightGBM",
    y_pred=y_pred_lgbm,
    y_prob=y_prob_lgbm,
    start_time=start_time,
    end_time=end_time
)