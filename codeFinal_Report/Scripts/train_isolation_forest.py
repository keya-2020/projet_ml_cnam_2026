# ==========================================================
# ISOLATION FOREST
# ==========================================================

import time
import numpy as np

from sklearn.ensemble import IsolationForest

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
print("ISOLATION FOREST TRAINING")
print("===================================================")

# ==========================================================
# MODEL
# ==========================================================

iso_model = IsolationForest(

    n_estimators=200,

    contamination=0.01,

    random_state=42,

    n_jobs=-1
)

# ==========================================================
# TRAINING
# ==========================================================

iso_model.fit(X_train_preprocessed)

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred_iso = iso_model.predict(
    X_test_preprocessed
)

# Convert:
# normal = 1
# anomaly = -1
# into:
# normal = 0
# fraud = 1

y_pred_iso = np.where(
    y_pred_iso == -1,
    1,
    0
)

# ==========================================================
# ANOMALY SCORES
# ==========================================================

y_scores_iso = -iso_model.decision_function(
    X_test_preprocessed
)

# ==========================================================
# END TIMER
# ==========================================================

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("\nIsolation Forest Results")

print(confusion_matrix(
    y_test,
    y_pred_iso
))

print(classification_report(
    y_test,
    y_pred_iso
))

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        y_scores_iso
    )
)

# ==========================================================
# SAVE
# ==========================================================

save_results(

    model_name="Isolation Forest",

    y_pred=y_pred_iso,

    y_prob=y_scores_iso,

    start_time=start_time,

    end_time=end_time
)

print("\nIsolation Forest completed!")