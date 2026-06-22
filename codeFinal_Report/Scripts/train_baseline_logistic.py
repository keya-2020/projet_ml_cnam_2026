# ==========================================================
# BASELINE LOGISTIC REGRESSION
# ==========================================================

from sklearn.linear_model import LogisticRegression
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

log_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

log_model.fit(
    X_train_preprocessed,
    y_train
)

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred_log = log_model.predict(
    X_test_preprocessed
)

y_prob_log = log_model.predict_proba(
    X_test_preprocessed
)[:,1]

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("Logistic Regression Results")

print(confusion_matrix(y_test, y_pred_log))

print(classification_report(
    y_test,
    y_pred_log
))

print(
    "ROC-AUC:",
    roc_auc_score(y_test, y_prob_log)
)

# ==========================================================
# SAVE
# ==========================================================

save_results(
    model_name="Logistic Regression",
    y_pred=y_pred_log,
    y_prob=y_prob_log,
    start_time=start_time,
    end_time=end_time
)