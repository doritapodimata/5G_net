import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np

df=pd.read_csv("new.csv")

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
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
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
num_bins = 5  # Χωρίζουμε το PathLoss σε 10 διαστήματα
y_binned = np.digitize(y, bins=np.histogram_bin_edges(y, bins=num_bins))

# Stratified K-Fold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
folds = list(skf.split(X, y_binned))
fold_sizes = [len(test_idx) for _, test_idx in folds]
print(fold_sizes)
