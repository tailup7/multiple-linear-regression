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

# CSV読み込み
df = pd.read_csv(file_path)

# 説明変数とするカラムを指定
feature_cols = ["MedInc", "HouseAge", "AveRooms"] 
# 目的変数とするカラムを指定
target_col   = ["MedInc"] 
t_input = [1, 0, 0, 0]

# csvから行列Xを作成
# データの個数(csvの行数)をN, 説明変数の数をMとして
# Xは N×(M+1) 行列で、1列目の成分はすべて1, 2列目以降はcsvの説明変数としたカラムを抜き出したもの
X_data = df[feature_cols].values
X = np.c_[np.ones(X_data.shape[0]), X_data]

print("選択されたファイル:", file_path)
print("Xの形状:", X.shape)
print(X[:5])  # 先頭5行表示

# 
# w = (((X^T)X)^(-1))X^T t
w = np.linalg.inv(X.T@X)@X.T@t_input