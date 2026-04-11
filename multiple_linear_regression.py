import pandas as pd
import tkinter as tk
from tkinter import filedialog
import numpy as np

root = tk.Tk()
root.withdraw()

# select input datafile(*.csv)
file_path = filedialog.askopenfilename(
    title="select csv",
    filetypes=[("csv file", "*.csv")]
)
if not file_path:
    print("No file was selected.")
    exit()

# load csv data
df = pd.read_csv(file_path)

# Specify feature columns (説明変数とするカラムを指定)
feature_cols = ["MedInc", "HouseAge", "AveRooms", "Latitude", "Longitude"] 
# Specify target column (目的変数とするカラムを指定)
target_col   = ["MedHouseValue"] 

y = df[target_col].values

# csvから行列Xを作成
# データの個数(csvの行数)をN, 説明変数の数をMとして
# Xは N×(M+1) 行列で、1列目の成分はすべて1, 2列目以降はcsvの説明変数としたカラムを抜き出したもの
X_raw = df[feature_cols].values
X = np.c_[np.ones(X_raw.shape[0]), X_raw]

# Randomly split the dataset into two equal halves
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42,
    shuffle=True
)

# Solve the normal equation using the training data (正規方程式)
# w = (((X^T)X)^(-1))X^T t
weights = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train

# Predict on the training data
y_train_pred = X_train @ weights

# Predict on the test data
y_test_pred = X_test @ weights

from sklearn.metrics import r2_score

# Evaluate with R^2 score
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print("Training R^2:", train_r2)
print("Test R^2:", test_r2)

import matplotlib.pyplot as plt

# Plot actual vs predicted values for the test data
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Test Data)")

# Plot the ideal reference line
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red")

plt.show()