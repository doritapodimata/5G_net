import numpy as np
import pandas as pd
from numpy import absolute
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
import optuna
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, cv, plot_importance
import seaborn as sns
from sklearn.inspection import permutation_importance


df = pd.read_csv("new_new.csv")

X = df.drop(columns=['PathLoss(db)'])  # select all columns except path-loss
y = df['PathLoss(db)']

# training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)


params = {'objective': 'reg:squarederror',
         'max_depth': 10,  # of the tree 4
         'alpha': 10,  # in case of overfiting
         'learning_rate': 0.08,  # slow learning
         'n_estimators': 73  # number of trees
         }
xgb_reg = XGBRegressor(**params)
xgb_reg.fit(X_train, y_train)


# test set predictions(smaller)
y_pred = xgb_reg.predict(X_test)
print(f"error rate for the test data", mean_absolute_error(y_test, y_pred))  # difference between prediction/actual


# trained set predictions
y_pred_train = xgb_reg.predict(X_train)
print("error rate for the trained data", mean_absolute_error(y_train, y_pred_train))

plot_importance(xgb_reg)
plt.title("Feature Importance (XGBoost Regressor)")
plt.rcParams['figure.figsize'] = [8, 4]
plt.show(),

# correlation matrix for explanation Feature Importance
correlation_matrix = df.corr()
print(correlation_matrix)

# Print heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

result = permutation_importance(xgb_reg, X_test, y_test, scoring='neg_mean_absolute_error', n_repeats=10, random_state=42)
perm_importance = pd.DataFrame({"Feature": X.columns, "Importance": result.importances_mean})
perm_importance = perm_importance.sort_values(by="Importance", ascending=False)

print(perm_importance)

# Feature importance από το XGBoost
importance = xgb_reg.feature_importances_
features = X.columns

# Correlation με το target variable
correlations = df.corr()["PathLoss(db)"].drop("PathLoss(db)")


# model evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)  # , as_pandas=True
scores = cross_val_score(xgb_reg, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print(scores[:5])

# make scores positive
scores = absolute(scores)
print(' MAE evaluation: %.3f (%.3f)' % (scores.mean(), scores.std()))


# optuna framework for tuning hyperparameters
def objective(trial):
   params_opt = {
       'objective': 'reg:squarederror',
       'n_estimators': trial.suggest_int('n_estimators', 50, 150),
       'max_depth': trial.suggest_int('max_depth', 3, 10),
       'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
       'alpha': trial.suggest_float('alpha', 0, 20),
       'lambda': trial.suggest_float('lambda', 0, 15),  # L2 regularization
   }

   model = XGBRegressor(**params_opt)

   # 10 fold cv named yolo because that's us
   yolo = cross_val_score(model, X_train, y_train,
                          scoring='neg_mean_absolute_error',
                          cv=10, n_jobs=-1)

   return -np.mean(yolo)  # makes the objective smaller


# run optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, timeout=60)  # 20 trials or 40 seconds


print("Best :", study.best_params)
print("Best CV MAE:", study.best_value)


# tarin final model with best params
best_model = XGBRegressor(**study.best_params)
best_model.fit(X_train, y_train)


# test set evaluation
y_pred = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print("Test MAE:", test_mae)


explainer = shap.Explainer(xgb_reg, X_train)
shap_values = explainer(X_test)


# visualize the predictions
shap.plots.beeswarm(shap_values)
shap.plots.waterfall(shap_values[5])

# Υπολογισμός MAPE
# def mean_absolute_percentage_error(y_true, y_pred):
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# mape = mean_absolute_percentage_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Test MAPE: {mape:.2f}%")
# print(f"Mean Squared Error (MSE): {mse:.4f}")
# print(f"Test R² Score: {r2:.3f}")

# std_y = np.std(df['PathLoss(db)'], ddof=1)  # ddof=1 για να υπολογιστεί σωστά για δείγμα
# print("Τυπικό σφάλμα του PathLoss:", std_y)

train_errors = []
test_errors = []

#τρεεσ
n_trees = range(10, 200, 10)

best_params = study.best_params.copy()
best_params.pop("n_estimators", None)

for n in n_trees:
    model = XGBRegressor(n_estimators=n, **best_params)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_errors.append(train_mae)
    test_errors.append(test_mae)


plt.figure(figsize=(8, 5))
plt.plot(n_trees, train_errors, label="Training Error", marker='o')
plt.plot(n_trees, test_errors, label="Test Error", marker='s')
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Overfitting Check: Train vs Test Error")
plt.legend()
plt.grid()
plt.show()