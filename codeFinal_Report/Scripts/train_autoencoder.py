# ==========================================================
# AUTOENCODER FRAUD DETECTION
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

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Dense
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
print("AUTOENCODER TRAINING")
print("===================================================")

# ==========================================================
# TRAIN ONLY ON NORMAL TRANSACTIONS
# ==========================================================

X_train_normal = X_train_preprocessed[
    y_train == 0
]

print(f"\nNormal training samples: {len(X_train_normal)}")

# ==========================================================
# INPUT DIMENSION
# ==========================================================

input_dim = X_train_normal.shape[1]

# ==========================================================
# MODEL
# ==========================================================

input_layer = Input(shape=(input_dim,))

# ==========================================================
# ENCODER
# ==========================================================

encoder = Dense(
    32,
    activation="relu"
)(input_layer)

encoder = Dense(
    16,
    activation="relu"
)(encoder)

encoder = Dense(
    8,
    activation="relu"
)(encoder)

# ==========================================================
# DECODER
# ==========================================================

decoder = Dense(
    16,
    activation="relu"
)(encoder)

decoder = Dense(
    32,
    activation="relu"
)(decoder)

decoder = Dense(
    input_dim,
    activation="linear"
)(decoder)

# ==========================================================
# AUTOENCODER MODEL
# ==========================================================

autoencoder = Model(
    inputs=input_layer,
    outputs=decoder
)

# ==========================================================
# COMPILE
# ==========================================================

autoencoder.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="mse"
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

history = autoencoder.fit(

    X_train_normal,

    X_train_normal,

    validation_split=0.2,

    epochs=30,

    batch_size=256,

    callbacks=[early_stop],

    verbose=1
)

# ==========================================================
# RECONSTRUCTION
# ==========================================================

X_test_reconstructed = autoencoder.predict(
    X_test_preprocessed
)

# ==========================================================
# RECONSTRUCTION ERROR
# ==========================================================

reconstruction_error = np.mean(

    np.square(
        X_test_preprocessed - X_test_reconstructed
    ),

    axis=1
)

# ==========================================================
# THRESHOLD
# ==========================================================

threshold = np.percentile(
    reconstruction_error,
    95
)

print(f"\nReconstruction Threshold: {threshold:.6f}")

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred_ae = (
    reconstruction_error >= threshold
).astype(int)

# ==========================================================
# END TIMER
# ==========================================================

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("\n===================================================")
print("AUTOENCODER RESULTS")
print("===================================================")

print("\nConfusion Matrix:")

print(confusion_matrix(
    y_test,
    y_pred_ae
))

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred_ae
))

print(
    "\nROC-AUC:",
    roc_auc_score(
        y_test,
        reconstruction_error
    )
)

# ==========================================================
# SAVE RESULTS
# ==========================================================

save_results(

    model_name="Autoencoder",

    y_pred=y_pred_ae,

    y_prob=reconstruction_error,

    start_time=start_time,

    end_time=end_time
)

print("\nAutoencoder completed!")