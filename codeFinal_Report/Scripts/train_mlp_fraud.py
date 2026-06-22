# ==========================================================
# MLP FRAUD DETECTION
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

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping
)

from tensorflow.keras.optimizers import Adam

from common_ml import *

# ==========================================================
# START TIMER
# ==========================================================

start_time = time.time()

print("===================================================")
print("MLP FRAUD TRAINING")
print("===================================================")

# ==========================================================
# MODEL
# ==========================================================

mlp_model = Sequential([

    # ------------------------------------------------------
    # INPUT LAYER
    # ------------------------------------------------------

    Dense(
        64,
        activation="relu",
        input_shape=(X_train_preprocessed.shape[1],)
    ),

    Dropout(0.3),

    # ------------------------------------------------------
    # HIDDEN LAYER
    # ------------------------------------------------------

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.3),

    # ------------------------------------------------------
    # HIDDEN LAYER
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
        learning_rate=0.001
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

    restore_best_weights=True
)

# ==========================================================
# TRAINING
# ==========================================================

history = mlp_model.fit(

    X_train_preprocessed,

    y_train,

    validation_split=0.2,

    epochs=30,

    batch_size=256,

    callbacks=[early_stop],

    class_weight={
        0: 1,
        1: 50
    },

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

threshold = 0.30

y_pred_mlp = (
    y_prob_mlp >= threshold
).astype(int)

# ==========================================================
# END TIMER
# ==========================================================

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("\n===================================================")
print("MLP RESULTS")
print("===================================================")

print("\nConfusion Matrix:")

print(confusion_matrix(
    y_test,
    y_pred_mlp
))

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred_mlp
))

print(
    "\nROC-AUC:",
    roc_auc_score(
        y_test,
        y_prob_mlp
    )
)

# ==========================================================
# SAVE RESULTS
# ==========================================================

save_results(

    model_name="MLP Deep Learning",

    y_pred=y_pred_mlp,

    y_prob=y_prob_mlp,

    start_time=start_time,

    end_time=end_time
)

print("\nMLP completed!")