import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models
import optuna

df = pd.read_csv("new_new.csv")

X = df.drop(columns=['PathLoss(db)'])  # select all columns except path-loss
y = df['PathLoss(db)']


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define Optuna objective
def objective(trial):
    # Hyperparameters to tune
    n_layers = trial.suggest_int("n_layers", 1, 3)
    units = trial.suggest_int("units", 16, 128)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2)

    def build_model():
        model = models.Sequential()
        model.add(layers.Input(shape=(X_scaled.shape[1],)))
        for _ in range(n_layers):
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1))  # Regression output
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_absolute_error')
        return model

    # 10-Fold CV
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    val_mae_scores = []

    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model()
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=50, batch_size=32, verbose=0)

        preds = model.predict(X_val).flatten()
        val_mae_scores.append(mean_absolute_error(y_val, preds))

    return np.mean(val_mae_scores)

# Run Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, timeout=600)

# Show best results
print("Best Parameters:", study.best_params)
print("Best MAE (10-fold CV):", study.best_value)
