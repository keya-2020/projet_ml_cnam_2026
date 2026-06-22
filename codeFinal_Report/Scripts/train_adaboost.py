# ==========================================================
# ADABOOST
# ==========================================================

from sklearn.ensemble import AdaBoostClassifier

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

ada_model = AdaBoostClassifier(

    n_estimators=150,
    learning_rate=0.05,

    random_state=42
)

ada_model.fit(
    X_train_preprocessed,
    y_train
)

# ==========================================================
# PREDICTIONS
# ==========================================================

# Fraud probabilities
y_prob_ada = ada_model.predict_proba(
    X_test_preprocessed
)[:, 1]

# ==========================================================
# THRESHOLD TUNING
# ==========================================================

threshold = 0.10

y_pred_ada = (
    y_prob_ada >= threshold
).astype(int)

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("AdaBoost Results")

print(confusion_matrix(
    y_test,
    y_pred_ada
))

print(classification_report(
    y_test,
    y_pred_ada
))

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        y_prob_ada
    )
)

# ==========================================================
# SAVE
# ==========================================================

save_results(
    model_name="AdaBoost",
    y_pred=y_pred_ada,
    y_prob=y_prob_ada,
    start_time=start_time,
    end_time=end_time
)