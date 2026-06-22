# ==========================================================
# MLP + SMOTE
# ==========================================================

import time
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Input
)

from tensorflow.keras.callbacks import (
    EarlyStopping
)

from tensorflow.keras.optimizers import Adam

from common_ml import *

# ==========================================================
# TIMER
# ==========================================================

start_time = time.time()

print("===================================================")
print("MLP + SMOTE TRAINING")
print("===================================================")

# ==========================================================
# BEFORE SMOTE
# ==========================================================

print("\nOriginal class distribution:")

print(
    pd.Series(y_train).value_counts()
)

# ==========================================================
# SMOTE
# ==========================================================

print("\nApplying SMOTE...")

smote = SMOTE(

    sampling_strategy=1.0,

    random_state=42,

    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(

    X_train_preprocessed,

    y_train
)

print("\nAfter SMOTE:")

print(
    pd.Series(y_train_smote).value_counts()
)

# ==========================================================
# MODEL
# ==========================================================

mlp_model = Sequential([

    Input(
        shape=(X_train_smote.shape[1],)
    ),

    # ------------------------------------------------------
    # BLOCK 1
    # ------------------------------------------------------

    Dense(128),

    BatchNormalization(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.30),

    # ------------------------------------------------------
    # BLOCK 2
    # ------------------------------------------------------

    Dense(64),

    BatchNormalization(),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.25),

    # ------------------------------------------------------
    # BLOCK 3
    # ------------------------------------------------------

    Dense(32),

    BatchNormalization(),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.20),

    # ------------------------------------------------------
    # BLOCK 4
    # ------------------------------------------------------

    Dense(
        16,
        activation="relu"
    ),

    # ------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------

    Dense(
        1,
        activation="sigmoid"
    )

])

# ==========================================================
# COMPILE
# ==========================================================

mlp_model.compile(

    optimizer=Adam(
        learning_rate=0.0005
    ),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

# ==========================================================
# EARLY STOPPING
# ==========================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)

# ==========================================================
# TRAIN
# ==========================================================

history = mlp_model.fit(

    X_train_smote,

    y_train_smote,

    validation_split=0.20,

    epochs=40,

    batch_size=512,

    callbacks=[early_stop],

    verbose=1
)

# ==========================================================
# PREDICTIONS
# ==========================================================

y_prob_mlp = mlp_model.predict(

    X_test_preprocessed

).flatten()

# ==========================================================
# THRESHOLD
# ==========================================================

threshold = 0.50

y_pred_mlp = (

    y_prob_mlp >= threshold

).astype(int)

# ==========================================================
# TIMER
# ==========================================================

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("\n===================================================")
print("MLP + SMOTE RESULTS")
print("===================================================")

cm = confusion_matrix(

    y_test,

    y_pred_mlp
)

print("\nConfusion Matrix:")

print(cm)

print("\nClassification Report:")

print(

    classification_report(

        y_test,

        y_pred_mlp
    )
)

roc_auc = roc_auc_score(

    y_test,

    y_prob_mlp
)

print("\nROC-AUC:", roc_auc)

print(

    "\nComputation Time:",

    round(
        end_time - start_time,
        2
    ),

    "seconds"
)

# ==========================================================
# SAVE RESULTS
# ==========================================================

save_results(

    model_name="MLP + SMOTE",

    y_pred=y_pred_mlp,

    y_prob=y_prob_mlp,

    start_time=start_time,

    end_time=end_time
)

# ==========================================================
# SAVE HISTORY
# ==========================================================

history_df = pd.DataFrame(

    history.history
)

history_df.to_csv(

    "MLP_SMOTE_history.csv",

    index=False
)

print("\nTraining history saved!")

# ==========================================================
# SAVE MODEL
# ==========================================================

mlp_model.save(

    "MLP_SMOTE.keras"
)

print("\nModel saved!")

# ==========================================================
# END
# ==========================================================

print("\n===================================================")
print("MLP + SMOTE COMPLETED")
print("===================================================")