# ==========================================================
# LOCAL OUTLIER FACTOR
# ==========================================================

import time
import numpy as np

from sklearn.neighbors import LocalOutlierFactor

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
print("LOCAL OUTLIER FACTOR TRAINING")
print("===================================================")

# ==========================================================
# MODEL
# ==========================================================

lof_model = LocalOutlierFactor(

    n_neighbors=20,

    contamination=0.01,

    novelty=True,

    n_jobs=-1
)

# ==========================================================
# TRAINING
# ==========================================================

lof_model.fit(X_train_preprocessed)

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred_lof = lof_model.predict(
    X_test_preprocessed
)

y_pred_lof = np.where(
    y_pred_lof == -1,
    1,
    0
)

# ==========================================================
# SCORES
# ==========================================================

y_scores_lof = -lof_model.decision_function(
    X_test_preprocessed
)

# ==========================================================
# END TIMER
# ==========================================================

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("\nLOF Results")

print(confusion_matrix(
    y_test,
    y_pred_lof
))

print(classification_report(
    y_test,
    y_pred_lof
))

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        y_scores_lof
    )
)

# ==========================================================
# SAVE
# ==========================================================

save_results(

    model_name="LOF",

    y_pred=y_pred_lof,

    y_prob=y_scores_lof,

    start_time=start_time,

    end_time=end_time
)

print("\nLOF completed!")