import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


df = pd.read_csv("new.csv")

# min-max normalization
scaler = MinMaxScaler()
values = df.columns.difference(["PathLoss(db)"])  # select all columns except path-loss
df[values] = scaler.fit_transform(df[values])
#df["PathLoss(db)"] = df["PathLoss(db)"]


num_bins = 7
binning = KBinsDiscretizer(num_bins, encode="ordinal", strategy="quantile")
df["PathLoss_binned"] = binning.fit_transform(df[["PathLoss(db)"]]).flatten()


#malakies
df.plot(kind='box', subplots=True, layout=(3, 3), figsize=(8, 8), sharex=False, sharey=False)
plt.suptitle("Box Plots for Numerical Columns")
plt.show()

#  10-Fold Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df[values], df["PathLoss_binned"])):
    print(f"Fold {fold_idx + 1}: Train size = {len(train_idx)}, Test size = {len(test_idx)}")

# bins distribution
print("\nBin distribution of PathLoss:")
print(df["PathLoss_binned"].value_counts().sort_index())

# new new dataset
df.to_csv("normalized_dataset.csv", index=False)
print("\nProcessed dataset saved as 'new_new.csv'")
