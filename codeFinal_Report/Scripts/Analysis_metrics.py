# ==========================================================
# METRICS.PY
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# LOAD RESULTS
# ==========================================================

results_df = pd.read_csv(
    "All_results.csv"
)

print("===================================================")
print("MODEL COMPARISON")
print("===================================================")

print(results_df)

# ==========================================================
# SORT BY ROC-AUC
# ==========================================================

results_df = results_df.sort_values(

    by="roc_auc",

    ascending=False
)

# ==========================================================
# SAVE SUMMARY
# ==========================================================

results_df.to_csv(

    "Model_Comparison.csv",

    index=False
)

# ==========================================================
# ROC-AUC BARPLOT
# ==========================================================

plt.figure(figsize=(10,6))

plt.bar(

    results_df["model_name"],

    results_df["roc_auc"]
)

plt.xticks(rotation=45)

plt.ylabel("ROC-AUC")

plt.title("Model ROC-AUC Comparison")

plt.tight_layout()

plt.show()

# ==========================================================
# F2-SCORE BARPLOT
# ==========================================================

plt.figure(figsize=(10,6))

plt.bar(

    results_df["model_name"],

    results_df["f2_score"]
)

plt.xticks(rotation=45)

plt.ylabel("F2-Score")

plt.title("Model F2-Score Comparison")

plt.tight_layout()

plt.show()

print("\nMetrics analysis completed!")