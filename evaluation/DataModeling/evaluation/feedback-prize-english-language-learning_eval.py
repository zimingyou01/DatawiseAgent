import os.path

import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--answer_file', type=str, required=True)
parser.add_argument('--predict_file', type=str, required=True)
parser.add_argument('--value', type=str, default="place_id")

args = parser.parse_args()

actual = pd.read_csv(args.answer_file)
submission = pd.read_csv(args.predict_file)

# Align rows on the submission key: the column whose values are unique and
# identical in both files, so sorting both by it pairs each row with its answer
key = None
for col in actual.columns:
    if col in submission.columns and actual[col].is_unique and submission[col].is_unique:
        left = actual[col].sort_values().reset_index(drop=True)
        right = submission[col].sort_values().reset_index(drop=True)
        numeric = pd.to_numeric(left, errors="coerce")
        if left.equals(right) or (
            numeric.notna().all() and numeric.equals(pd.to_numeric(right, errors="coerce"))
        ):
            key = col
            break
if key is not None:
    actual = actual.sort_values(by=key, ignore_index=True)
    submission = submission.sort_values(by=key, ignore_index=True)

def mcrmse(y_true, y_pred):
    """
    计算Mean Columnwise Root Mean Squared Error (MCRMSE)
    """
    assert y_true.shape == y_pred.shape, "The shapes of true and predicted values do not match"
    columnwise_rmse = np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))
    return columnwise_rmse.mean()

# 目标列 = 除键列外的所有列，避免依赖提交文件的列顺序
target_columns = [c for c in actual.columns if c != key] if key is not None else list(actual.columns[1:])
actual_values = actual[target_columns].values
predicted_values = submission[target_columns].values

# 计算 MCRMSE
performance = mcrmse(actual_values, predicted_values)

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
