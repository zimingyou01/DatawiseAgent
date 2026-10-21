import os.path

import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--answer_file', type=str, required=True)
parser.add_argument('--predict_file', type=str, required=True)

parser.add_argument('--value', type=str, default="Pastry,Z_Scratch,K_Scatch,Stains,Dirtiness,Bumps,Other_Faults")

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

# 逐列计算 ROC AUC，再取平均
target_columns = [c.strip() for c in args.value.split(",") if c.strip()]

performance = sum(
    roc_auc_score(actual[c], submission[c]) for c in target_columns
) / len(target_columns)

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
