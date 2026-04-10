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
    print("ファイルが選択されませんでした")
    exit()

df = pd.read_csv(file_path)

# 説明変数とするカラムを指定
feature_cols = ["MedInc", "HouseAge", "AveRooms", "Latitude", "Longitude"] 
# 目的変数とするカラムを指定
target_col   = ["MedHouseValue"] 

t_input = df[target_col].values

# csvから行列Xを作成
# データの個数(csvの行数)をN, 説明変数の数をMとして
# Xは N×(M+1) 行列で、1列目の成分はすべて1, 2列目以降はcsvの説明変数としたカラムを抜き出したもの
X_data = df[feature_cols].values
X = np.c_[np.ones(X_data.shape[0]), X_data]

# 正規方程式
# w = (((X^T)X)^(-1))X^T t
w = np.linalg.pinv(X.T @ X) @ X.T @ t_input

# 予測値
y_pred = X @ w

from sklearn.metrics import r2_score

# 決定係数 R^2 で評価
score = r2_score(t_input, y_pred)
print("R^2:", score)

import matplotlib.pyplot as plt

plt.scatter(t_input, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.plot([t_input.min(), t_input.max()],
         [t_input.min(), t_input.max()],
         color='red')  # 理想線
plt.show()