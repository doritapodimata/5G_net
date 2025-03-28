from sklearn.metrics import mean_absolute_error
import numpy as np
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("new_new.csv")

X = df.drop(columns=['PathLoss(db)'])  # all columns except PathLoss
y = df['PathLoss(db)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

#  Model: Mean Predictor
mean_prediction = np.full(shape=y_test.shape, fill_value=y_train.mean())
mae_mean = mean_absolute_error(y_test, mean_prediction)

#  Model: Median Predictor
median_prediction = np.full(shape=y_test.shape, fill_value=y_train.median())
mae_median = mean_absolute_error(y_test, median_prediction)


# Print Results
print(f"📊 Mean Predictor MAE:   {mae_mean}")
print(f"📊 Median Predictor MAE: {mae_median}")

# Check if XGBoost is significantly better
