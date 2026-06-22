# ==========================================================
# VOTING ENSEMBLE
# ==========================================================

import time

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

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
print("VOTING ENSEMBLE TRAINING")
print("===================================================")

# ==========================================================
# BASE MODELS
# ==========================================================

log_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

rf_model = RandomForestClassifier(

    n_estimators=200,

    max_depth=12,

    class_weight="balanced",

    n_jobs=-1,

    random_state=42
)

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

cat_model = CatBoostClassifier(

    iterations=200,

    depth=6,

    learning_rate=0.05,

    loss_function="Logloss",

    verbose=0,

    random_state=42
)

# ==========================================================
# VOTING MODEL
# ==========================================================

voting_model = VotingClassifier(

    estimators=[

        ("logistic", log_model),

        ("rf", rf_model),

        ("xgb", xgb_model),

        ("catboost", cat_model)

    ],

    voting="soft",

    n_jobs=-1
)

# ==========================================================
# TRAINING
# ==========================================================

voting_model.fit(
    X_train_preprocessed,
    y_train
)

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred_vote = voting_model.predict(
    X_test_preprocessed
)

y_prob_vote = voting_model.predict_proba(
    X_test_preprocessed
)[:,1]

# ==========================================================
# END TIMER
# ==========================================================

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("\nVoting Ensemble Results")

print(confusion_matrix(
    y_test,
    y_pred_vote
))

print(classification_report(
    y_test,
    y_pred_vote
))

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        y_prob_vote
    )
)

# ==========================================================
# SAVE
# ==========================================================

save_results(

    model_name="Voting Ensemble",

    y_pred=y_pred_vote,

    y_prob=y_prob_vote,

    start_time=start_time,

    end_time=end_time
)

print("\nVoting Ensemble completed!")