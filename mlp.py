import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import shap

# Load data
df = pd.read_csv("new_new.csv")
X = df.drop(columns=['PathLoss(db)', 'PathLoss_binned'], errors='ignore')
y = df['PathLoss(db)']


# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Optuna objective function
def objective(trial):
    # Hyperparameter space
    n_layers = trial.suggest_int("n_layers", 1, 3)
    units = trial.suggest_int("units", 16, 128)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2)

    # Build model inside the objective (uses trial params)
    def build_model():
        model = models.Sequential()
        model.add(layers.Input(shape=(X_scaled.shape[1],)))
        for _ in range(n_layers):
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1))  # regression output
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_absolute_error')
        return model

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=0
    )

    # KFold CV (not repeated)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    val_mae_scores = []

    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model()
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=50,
                  batch_size=32,
                  callbacks=[early_stop],  # ✅ Plugged in here
                  verbose=0)

        preds = model.predict(X_val).flatten()
        val_mae_scores.append(mean_absolute_error(y_val, preds))

    return np.mean(val_mae_scores)


# Run Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, timeout= 700) # 12 minutes

#  Train final model using best params
best = study.best_params

final_model = models.Sequential()
final_model.add(layers.Input(shape=(X_scaled.shape[1],)))
for _ in range(best['n_layers']):
    final_model.add(layers.Dense(best['units'], activation='relu'))
    final_model.add(layers.Dropout(best['dropout']))
final_model.add(layers.Dense(1))  # output

final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=best['lr']),
    loss='mean_absolute_error'
)

# early stopping for final training
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# 2. Fit model on full dataset (or split if you want)
final_model.fit(
    X_scaled, y,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


# small sample
background = X_scaled[np.random.choice(X_scaled.shape[0], 100, replace=False)]

#  DeepExplainer
explainer = shap.DeepExplainer(final_model, background)

# Explain a subset of your data
# sample_to_explain = X_scaled[:100]
shap_values = explainer.shap_values(X_scaled[:100])

# visualization
# shap.summary_plot(shap_values[0], sample_to_explain, feature_names=X.columns)
shap.summary_plot(shap_values[0], X_scaled[:100], feature_names=X.columns)

# Results
print(" Best Parameters:", study.best_params)
print(" Best MAE (10-fold CV):", study.best_value)
