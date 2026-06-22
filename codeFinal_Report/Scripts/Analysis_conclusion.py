# ==========================================================
# FRAUD DETECTION - ADVANCED MODEL COMPARISON DASHBOARD
# ==========================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch

try:
    from adjustText import adjust_text
    ADJUST_TEXT = True
except:
    ADJUST_TEXT = False
    
# ==========================================================
# STYLE
# ==========================================================

plt.style.use("ggplot")

sns.set_theme(
    style="whitegrid",
    context="talk"
)

# ==========================================================
# LOAD RESULTS
# ==========================================================

df = pd.read_csv("All_results.csv")

# ==========================================================
# MODEL FAMILY CLASSIFICATION
# ==========================================================

def get_family(model):

    model = str(model).lower()

    if any(x in model for x in [
        "lstm",
        "transformer",
        "cnn",
        "rnn",
        "autoencoder",
        "deep"
    ]):
        return "Deep Learning"

    elif any(x in model for x in [
        "voting",
        "stacking",
        "bagging",
        "blending",
        "adaboost",
        "gradient",
        "random forest",
        "extra trees",
        "xgboost",
        "catboost"
    ]):
        return "Ensemble"

    elif any(x in model for x in [
        "isolation",
        "lof",
        "oneclass",
        "dbscan"
    ]):
        return "Anomaly Detection"

    else:
        return "Supervised ML"


df["family"] = df["model_name"].apply(get_family)

# ==========================================================
# COLOR MAP
# ==========================================================

family_colors = {

    "Supervised ML": "#4C72B0",
    "Ensemble": "#55A868",
    "Deep Learning": "#C44E52",
    "Anomaly Detection": "#DD8452"
}

legend_elements = [

    Patch(
        facecolor=family_colors["Supervised ML"],
        label="Supervised ML"
    ),

    Patch(
        facecolor=family_colors["Ensemble"],
        label="Ensemble Methods"
    ),

    Patch(
        facecolor=family_colors["Deep Learning"],
        label="Deep Learning"
    ),

    Patch(
        facecolor=family_colors["Anomaly Detection"],
        label="Anomaly Detection"
    )

]

colors = df["family"].map(family_colors)

# ==========================================================
# SORTING
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
# ROC-AUC RANKING
# ==========================================================

fig, ax = plt.subplots(
    figsize=(15,10)
)

bars = ax.barh(
    df_auc["model_name"],
    df_auc["roc_auc"],
    color=df_auc["family"].map(family_colors)
)

ax.set_title(
    "ROC-AUC Ranking",
    fontsize=18,
    weight="bold"
)

ax.set_xlabel("ROC-AUC")

ax.invert_yaxis()

for i, v in enumerate(df_auc["roc_auc"]):

    ax.text(
        v + 0.002,
        i,
        f"{v:.3f}",
        va="center"
    )

ax.grid(
    axis="x",
    alpha=0.3
)

ax.legend(
    handles=legend_elements,
    loc="lower right",
    fontsize=10
)

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 2
# F2 SCORE RANKING
# ==========================================================

fig, ax = plt.subplots(
    figsize=(15,10)
)

bars = ax.barh(
    df_f2["model_name"],
    df_f2["f2_score"],
    color=df_f2["family"].map(family_colors)
)

ax.set_title(
    "Fraud Detection Ranking (F2 Score)",
    fontsize=18,
    weight="bold"
)

ax.set_xlabel("F2 Score")

ax.invert_yaxis()

for i, v in enumerate(df_f2["f2_score"]):

    ax.text(
        v + 0.002,
        i,
        f"{v:.3f}",
        va="center"
    )

ax.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 3
# PRECISION / RECALL
# ==========================================================

df_pr = df.sort_values(
    "recall",
    ascending=False
)

x = np.arange(len(df_pr))

width = 0.4

fig, ax = plt.subplots(
    figsize=(18,8)
)

ax.bar(
    x - width/2,
    df_pr["recall"],
    width,
    label="Recall"
)

ax.bar(
    x + width/2,
    df_pr["precision"],
    width,
    label="Precision"
)

ax.set_xticks(x)

ax.set_xticklabels(
    df_pr["model_name"],
    rotation=45,
    ha="right"
)

ax.set_ylim(0,1)

ax.set_title(
    "Precision vs Recall",
    fontsize=18,
    weight="bold"
)

ax.legend()

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 4
# F1 VS F2
# ==========================================================

df_f = df.sort_values(
    "f2_score",
    ascending=False
)

x = np.arange(len(df_f))

fig, ax = plt.subplots(
    figsize=(18,8)
)

ax.bar(
    x - width/2,
    df_f["f1_score"],
    width,
    label="F1"
)

ax.bar(
    x + width/2,
    df_f["f2_score"],
    width,
    label="F2"
)

ax.set_xticks(x)

ax.set_xticklabels(
    df_f["model_name"],
    rotation=45,
    ha="right"
)

ax.set_ylim(0, 0.4)

ax.set_yticks(
    np.arange(0, 0.41, 0.05)
)

ax.set_title(
    "F1 Score vs F2 Score",
    fontsize=18,
    weight="bold"
)

ax.legend()

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 5
# PERFORMANCE VS COMPUTATION COST
# ==========================================================

fig, ax = plt.subplots(
    figsize=(14,10)
)

for family in df["family"].unique():

    subset = df[
        df["family"] == family
    ]

    ax.scatter(
        subset["computation_time_sec"],
        subset["f2_score"],
        s=450,
        label=family
    )

texts = []

for _, row in df.iterrows():

    texts.append(

        ax.text(
            row["computation_time_sec"],
            row["f2_score"],
            row["model_name"],
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="gray",
                alpha=0.8
            )
        )

    )

if ADJUST_TEXT:
    adjust_text(
        texts,
        arrowprops=dict(
            arrowstyle="-",
            color="gray"
        )
    )
ax.set_xlabel(
    "Computation Time (seconds)"
)

ax.set_ylabel(
    "F2 Score"
)

ax.set_title(
    "Performance vs Computation Cost",
    fontsize=18,
    weight="bold"
)

ax.legend()

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 6
# ROC-AUC VS F2
# ==========================================================

fig, ax = plt.subplots(
    figsize=(14,10)
)

for family in df["family"].unique():

    subset = df[
        df["family"] == family
    ]

    ax.scatter(
        subset["roc_auc"],
        subset["f2_score"],
        s=250,
        label=family
    )

for _, row in df.iterrows():

    ax.annotate(
        row["model_name"],
        (
            row["roc_auc"],
            row["f2_score"]
        ),
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="white",
            edgecolor="gray",
            alpha=0.85
        )
    )

ax.axvline(
    0.80,
    linestyle="--",
    alpha=0.5
)

ax.set_xlabel(
    "ROC-AUC"
)

ax.set_ylabel(
    "F2 Score"
)

ax.set_title(
    "ROC-AUC vs F2 Score",
    fontsize=18,
    weight="bold"
)

ax.legend()

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 7
# PRECISION RECALL MAP
# ==========================================================

fig, ax = plt.subplots(
    figsize=(14,10)
)

for family in df["family"].unique():

    subset = df[
        df["family"] == family
    ]

    ax.scatter(
        subset["precision"],
        subset["recall"],
        s=250,
        label=family
    )

for _, row in df.iterrows():

    ax.annotate(
        row["model_name"],
        (
            row["precision"],
            row["recall"]
        ),
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="white",
            edgecolor="gray",
            alpha=0.85
        )
    )

ax.set_xlabel(
    "Precision"
)

ax.set_ylabel(
    "Recall"
)

ax.set_title(
    "Precision vs Recall Trade-off",
    fontsize=18,
    weight="bold"
)

ax.legend()

plt.tight_layout()
plt.show()


# ==========================================================
# FIGURE 8
# HEATMAP OF METRICS
# ==========================================================

metrics = [
    "precision",
    "recall",
    "f1_score",
    "f2_score",
    "roc_auc"
]

heatmap_data = (
    df
    .set_index("model_name")[metrics]
    .sort_values(
        "f2_score",
        ascending=False
    )
)

plt.figure(
    figsize=(18,14)
)

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={"label":"Performance Score"}
)

plt.title(
    "Model Performance Heatmap",
    fontsize=18,
    weight="bold"
)

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 9
# FAMILY COMPARISON
# ==========================================================

family_summary = (

    df.groupby("family")[
        [
            "roc_auc",
            "f2_score",
            "recall",
            "precision"
        ]
    ]

    .mean()

    .sort_values(
        "f2_score",
        ascending=False
    )
)

ax = family_summary.plot(
    kind="bar",
    figsize=(14,8),
    width=0.8
)

ax.legend(
    fontsize=10,
    loc="best"
)

ax.grid(
    axis="y",
    alpha=0.3
)

plt.title(
    "Average Performance by Model Family",
    fontsize=18,
    weight="bold"
)

plt.ylabel("Score")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()

# ==========================================================
# TOP MODELS
# ==========================================================

print("\n")
print("="*80)
print("TOP 10 MODELS BY F2 SCORE")
print("="*80)

print(

    df.sort_values(
        "f2_score",
        ascending=False
    )[[
        "model_name",
        "family",
        "f2_score",
        "recall",
        "precision",
        "roc_auc",
        "computation_time_sec"
    ]].head(10)

)

print("\n")
print("="*80)
print("TOP 10 MODELS BY ROC-AUC")
print("="*80)

print(

    df.sort_values(
        "roc_auc",
        ascending=False
    )[[
        "model_name",
        "family",
        "roc_auc",
        "f2_score",
        "recall",
        "precision"
    ]].head(10)

)