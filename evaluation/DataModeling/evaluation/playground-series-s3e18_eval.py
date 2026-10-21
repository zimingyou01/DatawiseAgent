import os.path

import pandas as pd
import argparse

from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--answer_file', type=str, required=True)
parser.add_argument('--predict_file', type=str, required=True)

parser.add_argument('--value', type=str, default="EC1,EC2")

args = parser.parse_args()

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

target_columns = [c.strip() for c in args.value.split(",") if c.strip()]

# Compute mean ROC AUC over the target columns
performance = sum(roc_auc_score(answers[c], predictions[c]) for c in target_columns) / len(target_columns)

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
