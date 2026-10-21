import os.path

import pandas as pd
import argparse
from sklearn.metrics import log_loss


parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--answer_file', type=str, required=True)
parser.add_argument('--predict_file', type=str, required=True)

parser.add_argument('--value', type=str, default="winner_model_a,winner_model_b,winner_tie")

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

# 提取实际值和预测值（列顺序由 --value 指定）
target_columns = [c.strip() for c in args.value.split(",") if c.strip()]
actual_values = actual[target_columns].values
predicted_values = submission[target_columns].values


performance = log_loss(actual_values, predicted_values)


with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
