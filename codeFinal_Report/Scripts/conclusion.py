# ==========================================================
# FRAUD DETECTION - MODEL COMPARISON DASHBOARD
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# LOAD RESULTS
# ==========================================================

df = pd.read_csv("All_results.csv")

# ==========================================================
# SORT DATASETS
# ==========================================================

df_auc = df.sort_values(
    "roc_auc",
    ascending=False
)

df_f2 = df.sort_values(
    "f2_score",
    ascending=False
)

# ==========================================================
# FIGURE 1
# ROC-AUC COMPARISON
# ==========================================================

plt.figure(figsize=(12,6))

plt.barh(
    df_auc["model_name"],
    df_auc["roc_auc"]
)

plt.xlabel("ROC-AUC")
plt.title("ROC-AUC Ranking")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 2
# F2-SCORE COMPARISON
# ==========================================================

plt.figure(figsize=(12,6))

plt.barh(
    df_f2["model_name"],
    df_f2["f2_score"]
)

plt.xlabel("F2 Score")

plt.title(
    "Fraud Detection Ranking (F2 Score)"
)

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 3
# RECALL VS PRECISION
# ==========================================================

plt.figure(figsize=(12,6))

x = range(len(df))

width = 0.4

plt.bar(

    [i-width/2 for i in x],

    df["recall"],

    width=width,

    label="Recall"
)

plt.bar(

    [i+width/2 for i in x],

    df["precision"],

    width=width,

    label="Precision"
)

plt.xticks(

    x,

    df["model_name"],

    rotation=70
)

plt.ylabel("Score")

plt.title(
    "Recall vs Precision"
)

plt.legend()

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 4
# F1 VS F2
# ==========================================================

plt.figure(figsize=(12,6))

x = range(len(df))

width = 0.4

plt.bar(

    [i-width/2 for i in x],

    df["f1_score"],

    width=width,

    label="F1"
)

plt.bar(

    [i+width/2 for i in x],

    df["f2_score"],

    width=width,

    label="F2"
)

plt.xticks(

    x,

    df["model_name"],

    rotation=70
)

plt.ylabel("Score")

plt.title(
    "F1 Score vs F2 Score"
)

plt.legend()

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 5
# PERFORMANCE VS COMPUTATION COST
# ==========================================================

plt.figure(figsize=(10,8))

plt.scatter(

    df["computation_time_sec"],

    df["f2_score"],

    s=150
)

for _, row in df.iterrows():

    plt.annotate(

        row["model_name"],

        (
            row["computation_time_sec"],
            row["f2_score"]
        ),

        fontsize=8
    )

plt.xlabel(
    "Computation Time (seconds)"
)

plt.ylabel(
    "F2 Score"
)

plt.title(
    "Fraud Detection Efficiency"
)

plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 6
# ROC-AUC VS F2
# ==========================================================

plt.figure(figsize=(10,8))

plt.scatter(

    df["roc_auc"],

    df["f2_score"],

    s=180
)

for _, row in df.iterrows():

    plt.annotate(

        row["model_name"],

        (
            row["roc_auc"],
            row["f2_score"]
        ),

        fontsize=8
    )

plt.xlabel(
    "ROC-AUC"
)

plt.ylabel(
    "F2 Score"
)

plt.title(
    "Fraud Detection Performance Map"
)

plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 7
# PRECISION-RECALL MAP
# ==========================================================

plt.figure(figsize=(10,8))

plt.scatter(

    df["precision"],

    df["recall"],

    s=180
)

for _, row in df.iterrows():

    plt.annotate(

        row["model_name"],

        (
            row["precision"],
            row["recall"]
        ),

        fontsize=8
    )

plt.xlabel(
    "Precision"
)

plt.ylabel(
    "Recall"
)

plt.title(
    "Precision vs Recall Trade-off"
)

plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================================
# TOP MODELS BY F2
# ==========================================================

print("\n===================================================")
print("TOP FRAUD DETECTION MODELS (F2 SCORE)")
print("===================================================\n")

print(

    df.sort_values(

        "f2_score",

        ascending=False

    )[

        [
            "model_name",
            "f2_score",
            "recall",
            "precision",
            "roc_auc",
            "computation_time_sec"
        ]

    ].head(10)

)

# ==========================================================
# TOP MODELS BY ROC-AUC
# ==========================================================

print("\n===================================================")
print("TOP MODELS (ROC-AUC)")
print("===================================================\n")

print(

    df.sort_values(

        "roc_auc",

        ascending=False

    )[

        [
            "model_name",
            "roc_auc",
            "f2_score",
            "recall",
            "precision"
        ]

    ].head(10)

)