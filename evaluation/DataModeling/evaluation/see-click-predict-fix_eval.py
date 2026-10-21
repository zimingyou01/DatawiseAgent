import os.path

import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--answer_file', type=str, required=True)
parser.add_argument('--predict_file', type=str, required=True)

# 目标列，多个列用逗号分隔
parser.add_argument('--value', type=str, default="num_views,num_votes,num_comments")

args = parser.parse_args()


def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

# Align rows on the submission key: the column whose values are unique and
# identical in both files, so sorting both by it pairs each row with its answer
key = None
for col in answers.columns:
    if col in predictions.columns and answers[col].is_unique and predictions[col].is_unique:
        left = answers[col].sort_values().reset_index(drop=True)
        right = predictions[col].sort_values().reset_index(drop=True)
        numeric = pd.to_numeric(left, errors="coerce")
        if left.equals(right) or (
            numeric.notna().all() and numeric.equals(pd.to_numeric(right, errors="coerce"))
        ):
            key = col
            break
if key is not None:
    answers = answers.sort_values(by=key, ignore_index=True)
    predictions = predictions.sort_values(by=key, ignore_index=True)

target_columns = [col.strip() for col in args.value.split(",") if col.strip()]

# Compute RMSLE, averaged over all target columns
performance = np.mean(
    [rmsle(answers[col], predictions[col]) for col in target_columns]
)

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
