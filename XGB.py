import numpy as np
import pandas as pd
from numpy import absolute
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
import optuna
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, cv, plot_importance

df = pd.read_csv("new_new.csv")
# type
# df['Latitude'].dtype
# print(df.dtypes)

X = df.drop(columns=['PathLoss(db)'])  # select all columns except path-loss
y = df['PathLoss(db)']

# training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

params = {'objective': 'reg:squarederror',
          'max_depth': 4,  # of the tree
          'alpha': 10,  # in case of overfilling
          'learning_rate': 0.5,  # slow learning
          'n_estimators': 85  # number of trees
          }
xgb_reg = XGBRegressor(**params)
xgb_reg.fit(X_train, y_train)

# test set predictions
y_pred = xgb_reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"error for the test data", mae)  # difference between prediction and actual values

# trained set predictions
y_pred_train = xgb_reg.predict(X_train)
print("error for the trained data", mean_absolute_error(y_train, y_pred_train))

plot_importance(xgb_reg)
plt.title("Feature Importance (XGBoost Regressor)")
plt.rcParams['figure.figsize'] = [6, 4]
plt.show()

# model evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)  # , as_pandas=True
scores = cross_val_score(xgb_reg, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print(scores[:5])

# make scores positive
scores = absolute(scores)
print(' MAE evaluation: %.3f (%.3f)' % (scores.mean(), scores.std()))


# optuna framework
def objective(trial):
    params_opt = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'alpha': trial.suggest_float('alpha', 0, 10),
        'lambda': trial.suggest_float('lambda', 0, 10),  # L2 regularization
    }

    model = XGBRegressor(**params_opt)

    # 10 fold cv named yolo because thats us
    yolo = cross_val_score(model, X_train, y_train,
                           scoring='neg_mean_absolute_error',
                           cv=10, n_jobs=-1)

    return -np.mean(yolo)  # makes the objective smaller


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, timeout=60)  # 50 trials or 3,5 minutes

print("Best :", study.best_params)
print("Best CV MAE:", study.best_value)

# tarin final model with best params
best_model = XGBRegressor(**study.best_params)
best_model.fit(X_train, y_train)

# test set evaluation
y_pred = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print("Test MAE:", test_mae)

