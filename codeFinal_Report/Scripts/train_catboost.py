# ==========================================================
# CATBOOST
# ==========================================================

from catboost import CatBoostClassifier

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

cat_model = CatBoostClassifier(

    iterations=200,
    depth=6,
    learning_rate=0.05,

    loss_function="Logloss",

    verbose=0,
    random_state=42
)

cat_model.fit(
    X_train_preprocessed,
    y_train
)

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred_cat = cat_model.predict(
    X_test_preprocessed
)

y_prob_cat = cat_model.predict_proba(
    X_test_preprocessed
)[:,1]

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("CatBoost Results")

print(confusion_matrix(
    y_test,
    y_pred_cat
))

print(classification_report(
    y_test,
    y_pred_cat
))

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        y_prob_cat
    )
)

# ==========================================================
# SAVE
# ==========================================================

save_results(
    model_name="CatBoost",
    y_pred=y_pred_cat,
    y_prob=y_prob_cat,
    start_time=start_time,
    end_time=end_time
)