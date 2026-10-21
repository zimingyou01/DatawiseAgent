import os.path

import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--answer_file', type=str, required=True)
parser.add_argument('--predict_file', type=str, required=True)

parser.add_argument('--value', type=str, default="generated")

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

# 移除键列，剩下的是矩阵的值
drop_key = key if key is not None else actual.columns[0]
submission_values = submission.drop(columns=[drop_key]).values
actual_values = actual.drop(columns=[drop_key]).values

# 计算平均绝对误差
performance = np.mean(np.abs(submission_values - actual_values))


with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
