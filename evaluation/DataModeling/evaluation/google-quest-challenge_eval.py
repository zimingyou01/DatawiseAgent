import os.path

import pandas as pd
import argparse
from scipy.stats import spearmanr


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
def mean_spearmanr(y_true, y_pred):
    """
    计算每列的Spearman's rank correlation coefficient，并取平均值
    """
    assert y_true.shape == y_pred.shape, "The shapes of true and predicted values do not match"
    correlations = []
    for col in range(y_true.shape[1]):
        corr, _ = spearmanr(y_true[:, col], y_pred[:, col])
        correlations.append(corr)
    return sum(correlations) / len(correlations)


# 目标列 = 除键列外的所有列，避免依赖提交文件的列顺序
target_columns = [c for c in actual.columns if c != key] if key is not None else list(actual.columns[1:])
actual_values = actual[target_columns].values
predicted_values = submission[target_columns].values
# 计算平均 Spearman 相关系数
performance = mean_spearmanr(actual_values, predicted_values)


with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
