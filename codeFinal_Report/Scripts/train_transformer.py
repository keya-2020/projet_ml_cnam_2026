# ==========================================================
# FT-TRANSFORMER FRAUD DETECTION
# ==========================================================

# ==========================================================
# IMPORTS
# ==========================================================

import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from pytorch_tabular import TabularModel

from pytorch_tabular.models.ft_transformer.config import (
    FTTransformerConfig
)

from pytorch_tabular.config import (
    DataConfig,
    TrainerConfig,
    OptimizerConfig
)

from common_ml import *

# ==========================================================
# START TIMER
# ==========================================================

start_time = time.time()

print("===================================================")
print("FT-TRANSFORMER FRAUD TRAINING")
print("===================================================")

# ==========================================================
# FEATURE NAMES
# ==========================================================

feature_names = [

    f"F{i}"

    for i in range(
        X_train_preprocessed.shape[1]
    )
]

# ==========================================================
# TRAIN DATAFRAME
# ==========================================================

df_train = pd.DataFrame(

    X_train_preprocessed,

    columns=feature_names
)

df_train["target"] = y_train.values

# ==========================================================
# TEST DATAFRAME
# ==========================================================

df_test = pd.DataFrame(

    X_test_preprocessed,

    columns=feature_names
)

df_test["target"] = y_test.values

# ==========================================================
# DATA CONFIG
# ==========================================================

data_config = DataConfig(

    target=["target"],

    continuous_cols=feature_names,

    categorical_cols=[]
)

# ==========================================================
# MODEL CONFIG
# ==========================================================

model_config = FTTransformerConfig(

    task="classification",

    learning_rate=1e-3,

    input_embed_dim=32,

    num_heads=8,

    num_attn_blocks=4
)

# ==========================================================
# TRAINER CONFIG
# ==========================================================

trainer_config = TrainerConfig(

    max_epochs=15,

    accelerator="auto",

    batch_size=1024
)

# ==========================================================
# OPTIMIZER CONFIG
# ==========================================================

optimizer_config = OptimizerConfig()

# ==========================================================
# MODEL
# ==========================================================

ft_model = TabularModel(

    data_config=data_config,

    model_config=model_config,

    optimizer_config=optimizer_config,

    trainer_config=trainer_config
)

# ==========================================================
# TRAINING
# ==========================================================

ft_model.fit(

    train=df_train,

    validation=df_test
)

# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = ft_model.predict(

    df_test
)

# ==========================================================
# PROBABILITIES
# ==========================================================

prob_cols = [

    c

    for c in predictions.columns

    if "probability" in c.lower()
]

if len(prob_cols) == 0:

    raise ValueError(
        "Probability column not found."
    )

y_prob_ft = predictions[
    prob_cols[-1]
].values

# ==========================================================
# THRESHOLD
# ==========================================================

threshold = 0.30

y_pred_ft = (
    y_prob_ft >= threshold
).astype(int)

# ==========================================================
# END TIMER
# ==========================================================

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("\n===================================================")
print("FT-TRANSFORMER RESULTS")
print("===================================================")

print("\nConfusion Matrix:")

print(

    confusion_matrix(

        y_test,

        y_pred_ft
    )
)

print("\nClassification Report:")

print(

    classification_report(

        y_test,

        y_pred_ft
    )
)

print(

    "\nROC-AUC:",

    roc_auc_score(

        y_test,

        y_prob_ft
    )
)

# ==========================================================
# SAVE RESULTS
# ==========================================================

save_results(

    model_name="FT-Transformer",

    y_pred=y_pred_ft,

    y_prob=y_prob_ft,

    start_time=start_time,

    end_time=end_time
)

print("\nFT-Transformer completed!")