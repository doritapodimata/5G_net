import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold
import numpy as np

df = pd.read_csv("new.csv")

print(df.info())
print(df.columns)

numeric_cols = df.select_dtypes(include=['number'])
numeric_cols.plot(kind='box', subplots=True, layout=(3, 3), figsize=(8, 8), sharex=False, sharey=False)
plt.suptitle("Box Plots for Numerical Columns")
plt.show()

# plt.figure(figsize=(25,5))
# # plt.subplot(1,3,1)
# plt.title('Counter Plot')
# sns.countplot(x='PathLoss(db)',data=df)
# plt.show()

# Min-Max κανονικοποίηση
scaler = MinMaxScaler()
features = df.drop(columns=["PathLoss(db)"])  # Exclude target
df_normalized = pd.df(scaler.fit_transform(features), columns=features.columns)
df_normalized["PathLoss(db)"] = pd.df("Pathloss(db)")
print(df_normalized.head())

# print all the rows
# pd.set_option('display.max_rows', None)
# print(df_normalized)

# save normalized dataset to a new CSV
# normalized_file_path = "normalized_dataset.csv"
# df_normalized.to_csv(normalized_file_path, index=False)
# print(normalized_file_path)

# features (X) και target (y)
X = df.drop(columns=["PathLoss(db)"])  # oλες οι μεταβλητές εκτός από το target
y = df["PathLoss(db)"]

# Stratified K-Fold Cross Validation
# create bins for stratification (διακριτοποίηση συνεχούς μεταβλητής PathLoss)
num_bins = 7  # Χωρίζουμε το PathLoss σε 7 διαστήματα

#binnig
binning = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
y_binned = binning.fit_transform(y.values.reshape(-1, 1)).flatten()

# Stratified K-Fold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# folds = list(skf.split(X, y_binned))
# fold_sizes = [len(test_idx) for _, test_idx in folds]
# print(fold_sizes)

# Έλεγχος της κατανομής των bins
print("Κατανομή των bins:")
print(pd.Series(y_binned).value_counts().sort_index())

# Αποθήκευση των binned labels για μελλοντική χρήση
df['PathLoss_binned'] = y_binned


