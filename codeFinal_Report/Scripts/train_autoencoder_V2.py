# ==========================================================
# AUTOENCODER FRAUD DETECTION V2
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
# TIMER
# ==========================================================

start_time = time.time()

print("===================================================")
print("AUTOENCODER V2 TRAINING")
print("===================================================")

# ==========================================================
# NORMAL TRANSACTIONS ONLY
# ==========================================================

X_train_normal = X_train_preprocessed[
    y_train == 0
]

print(
    f"\nNormal training samples: {len(X_train_normal)}"
)

# ==========================================================
# INPUT DIMENSION
# ==========================================================

input_dim = X_train_normal.shape[1]

# ==========================================================
# INPUT
# ==========================================================

input_layer = Input(
    shape=(input_dim,)
)

# ==========================================================
# ENCODER
# ==========================================================

x = Dense(
    64,
    activation="relu"
)(input_layer)

x = BatchNormalization()(x)

x = Dropout(0.20)(x)

x = Dense(
    32,
    activation="relu"
)(x)

x = BatchNormalization()(x)

x = Dropout(0.15)(x)

latent = Dense(
    16,
    activation="relu"
)(x)

# ==========================================================
# DECODER
# ==========================================================

x = Dense(
    32,
    activation="relu"
)(latent)

x = BatchNormalization()(x)

x = Dense(
    64,
    activation="relu"
)(x)

output_layer = Dense(
    input_dim,
    activation="linear"
)(x)

# ==========================================================
# MODEL
# ==========================================================

autoencoder = Model(

    inputs=input_layer,

    outputs=output_layer
)

# ==========================================================
# COMPILE
# ==========================================================

autoencoder.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="mae"
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

history = autoencoder.fit(

    X_train_normal,

    X_train_normal,

    validation_split=0.20,

    epochs=50,

    batch_size=256,

    callbacks=[
        early_stop
    ],

    verbose=1
)

# ==========================================================
# TRAIN RECONSTRUCTION ERROR
# ==========================================================

X_train_reconstructed = autoencoder.predict(
    X_train_normal
)

train_error = np.sum(

    np.square(

        X_train_normal
        -
        X_train_reconstructed

    ),

    axis=1
)

# ==========================================================
# THRESHOLD
# ==========================================================

threshold = np.percentile(

    train_error,

    99
)

print(
    f"\nReconstruction Threshold: {threshold:.6f}"
)

# ==========================================================
# TEST RECONSTRUCTION
# ==========================================================

X_test_reconstructed = autoencoder.predict(

    X_test_preprocessed
)

# ==========================================================
# ANOMALY SCORE
# ==========================================================

reconstruction_error = np.sum(

    np.square(

        X_test_preprocessed
        -
        X_test_reconstructed

    ),

    axis=1
)

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred_ae = (

    reconstruction_error >= threshold

).astype(int)

# ==========================================================
# TIMER
# ==========================================================

end_time = time.time()

# ==========================================================
# RESULTS
# ==========================================================

print("\n===================================================")
print("AUTOENCODER V2 RESULTS")
print("===================================================")

cm = confusion_matrix(

    y_test,

    y_pred_ae
)

print("\nConfusion Matrix:")

print(cm)

print("\nClassification Report:")

print(

    classification_report(

        y_test,

        y_pred_ae
    )
)

roc_auc = roc_auc_score(

    y_test,

    reconstruction_error
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

    model_name="Autoencoder V2",

    y_pred=y_pred_ae,

    y_prob=reconstruction_error,

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

    "Autoencoder_V2_History.csv",

    index=False
)

# ==========================================================
# SAVE MODEL
# ==========================================================

autoencoder.save(

    "Autoencoder_V2.keras"
)

print("\nTraining history saved!")
print("Model saved!")

# ==========================================================
# END
# ==========================================================

print("\n===================================================")
print("AUTOENCODER V2 COMPLETED")
print("===================================================")