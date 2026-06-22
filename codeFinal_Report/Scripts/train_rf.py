# ==========================================================
# TRAIN_RF.PY
# Random Forest Fraud Detection
# ==========================================================

# ==========================================================
# IMPORTS
# ==========================================================

import time

from sklearn.ensemble import RandomForestClassifier

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
print("RANDOM FOREST TRAINING")
print("===================================================")

# ==========================================================
# MODEL
# ==========================================================

rf_model = RandomForestClassifier(

    # Number of trees
    n_estimators=200,

    # Maximum tree depth
    max_depth=12,

    # Handle class imbalance
    class_weight="balanced",

    # Use all CPU cores
    n_jobs=-1,

    # Reproducibility
    random_state=42
)

# ==========================================================
# TRAINING
# ==========================================================

print("\nTraining Random Forest...")

rf_model.fit(
    X_train_preprocessed,
    y_train
)

print("Training completed!")

# ==========================================================
# PREDICTIONS
# ==========================================================

print("\nGenerating predictions...")

y_pred_rf = rf_model.predict(
    X_test_preprocessed
)

y_prob_rf = rf_model.predict_proba(
    X_test_preprocessed
)[:, 1]

# ==========================================================
# END TIMER
# ==========================================================

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("\n===================================================")
print("RANDOM FOREST RESULTS")
print("===================================================")

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred_rf
)

print("\nConfusion Matrix:")
print(cm)

# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_rf
    )
)

# ----------------------------------------------------------
# ROC-AUC
# ----------------------------------------------------------

roc_auc = roc_auc_score(
    y_test,
    y_prob_rf
)

print(f"\nROC-AUC: {roc_auc:.4f}")

# ----------------------------------------------------------
# Computation Time
# ----------------------------------------------------------

computation_time = end_time - start_time

print(f"\nComputation Time: {computation_time:.2f} seconds")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

print("\n===================================================")
print("FEATURE IMPORTANCE")
print("===================================================")

feature_importance = rf_model.feature_importances_

importance_df = pd.DataFrame({

    "feature": X.columns,
    "importance": feature_importance

})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print(importance_df.head(15))

# ==========================================================
# SAVE FEATURE IMPORTANCE
# ==========================================================

importance_df.to_csv(
    "rf_feature_importance.csv",
    index=False
)

print("\nFeature importance saved!")

# ==========================================================
# SAVE RESULTS
# ==========================================================

save_results(

    model_name="Random Forest",

    y_pred=y_pred_rf,

    y_prob=y_prob_rf,

    start_time=start_time,

    end_time=end_time
)

print("\n===================================================")
print("RANDOM FOREST PIPELINE COMPLETED")
print("===================================================")