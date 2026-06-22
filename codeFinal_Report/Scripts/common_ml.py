# ==========================================================
# COMMON_ML.PY
# ==========================================================

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    fbeta_score
)

# ==========================================================
# LOAD DATA
# ==========================================================

df_selected = pd.read_csv("data_selected_report.csv")

# ==========================================================
# FEATURES / TARGET
# ==========================================================

X = df_selected.drop("fraud_bool", axis=1)

y = df_selected["fraud_bool"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==========================================================
# SCALING
# ==========================================================

scaler = StandardScaler()

X_train_preprocessed = scaler.fit_transform(X_train)

X_test_preprocessed = scaler.transform(X_test)

# ==========================================================
# GLOBAL RESULTS
# ==========================================================

results = {}

# ==========================================================
# SAVE RESULTS FUNCTION
# ==========================================================

def save_results(
    model_name,
    y_pred,
    y_prob=None,
    start_time=None,
    end_time=None
):

    cm = confusion_matrix(y_test, y_pred).tolist()

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    accuracy = accuracy_score(y_test, y_pred)

    roc_auc = (
        roc_auc_score(y_test, y_prob)
        if y_prob is not None
        else None
    )

    f2_score = fbeta_score(
        y_test,
        y_pred,
        beta=2
    )

    computation_time = (
        end_time - start_time
        if start_time and end_time
        else None
    )

    results[model_name] = {
        "confusion_matrix": cm,
        "classification_report": report,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "f2_score": f2_score,
        "computation_time_sec": computation_time
    }

    summary_row = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "f2_score": f2_score,
        "roc_auc": roc_auc,
        "true_negative": cm[0][0],
        "false_positive": cm[0][1],
        "false_negative": cm[1][0],
        "true_positive": cm[1][1],
        "computation_time_sec": computation_time,
    }

    try:

        df_existing = pd.read_csv(
            "All_results.csv"
        )

        df_all = pd.concat(
            [
                df_existing,
                pd.DataFrame([summary_row])
            ],
            ignore_index=True
        )

    except FileNotFoundError:

        df_all = pd.DataFrame([summary_row])

    df_all.to_csv(
        "All_results.csv",
        index=False
    )

    print(f"\nResults saved for: {model_name}")