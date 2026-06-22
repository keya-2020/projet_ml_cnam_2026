# ==========================================================
# ONE CLASS SVM
# ==========================================================

import time
import numpy as np

from sklearn.svm import OneClassSVM

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from common_ml import *

# ==========================================================
# SAMPLE DATA
# ==========================================================

sample_size = 50000

X_train_sample = X_train_preprocessed[:sample_size]

# ==========================================================
# START TIMER
# ==========================================================

start_time = time.time()

print("===================================================")
print("ONE CLASS SVM TRAINING")
print("===================================================")

# ==========================================================
# MODEL
# ==========================================================

ocsvm_model = OneClassSVM(

    kernel="rbf",

    gamma="scale",

    nu=0.01
)

# ==========================================================
# TRAINING
# ==========================================================

ocsvm_model.fit(X_train_sample)

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred_ocsvm = ocsvm_model.predict(
    X_test_preprocessed
)

y_pred_ocsvm = np.where(
    y_pred_ocsvm == -1,
    1,
    0
)

# ==========================================================
# SCORES
# ==========================================================

y_scores_ocsvm = -ocsvm_model.decision_function(
    X_test_preprocessed
)

# ==========================================================
# END TIMER
# ==========================================================

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("\nOne-Class SVM Results")

print(confusion_matrix(
    y_test,
    y_pred_ocsvm
))

print(classification_report(
    y_test,
    y_pred_ocsvm
))

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        y_scores_ocsvm
    )
)

# ==========================================================
# SAVE
# ==========================================================

save_results(

    model_name="One-Class SVM",

    y_pred=y_pred_ocsvm,

    y_prob=y_scores_ocsvm,

    start_time=start_time,

    end_time=end_time
)

print("\nOne-Class SVM completed!")