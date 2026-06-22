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

from sklearn.utils.class_weight import (
    compute_class_weight
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization
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
# CLASS WEIGHTS
# ==========================================================

classes = np.unique(y_train)

weights = compute_class_weight(

    class_weight="balanced",

    classes=classes,

    y=y_train
)

class_weights = {

    0: float(weights[0]),

    1: float(weights[1])
}

print("\nClass Weights:")
print(class_weights)

# ==========================================================
# MODEL
# ==========================================================

mlp_model = Sequential([

    # ------------------------------------------------------
    # INPUT
    # ------------------------------------------------------

    Input(
        shape=(X_train_preprocessed.shape[1],)
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

    Dropout(0.40),

    # ------------------------------------------------------
    # BLOCK 2
    # ------------------------------------------------------

    Dense(64),

    BatchNormalization(),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.30),

    # ------------------------------------------------------
    # BLOCK 3
    # ------------------------------------------------------

    Dense(32),

    BatchNormalization(),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.25),

    # ------------------------------------------------------
    # BLOCK 4
    # ------------------------------------------------------

    Dense(
        16,
        activation="relu"
    ),

    Dropout(0.20),

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

    metrics=[
        "accuracy"
    ]
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
# TRAINING
# ==========================================================

history = mlp_model.fit(

    X_train_preprocessed,

    y_train,

    validation_split=0.20,

    epochs=40,

    batch_size=256,

    callbacks=[
        early_stop
    ],

    class_weight=class_weights,

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
print(
    confusion_matrix(
        y_test,
        y_pred_mlp
    )
)

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
    round(end_time - start_time, 2),
    "seconds"
)

# ==========================================================
# SAVE RESULTS
# ==========================================================

save_results(

    model_name="MLP Deep Learning v2",

    y_pred=y_pred_mlp,

    y_prob=y_prob_mlp,

    start_time=start_time,

    end_time=end_time
)

# ==========================================================
# SAVE TRAINING HISTORY
# ==========================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    "MLP_training_history.csv",
    index=False
)

print("\nTraining history saved!")

# ==========================================================
# SAVE MODEL
# ==========================================================

mlp_model.save(
    "MLP_Fraud_Model.keras"
)

print("Model saved!")

# ==========================================================
# END
# ==========================================================

print("\n===================================================")
print("MLP PIPELINE COMPLETED")
print("===================================================")