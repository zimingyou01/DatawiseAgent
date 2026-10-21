import os.path

import pandas as pd
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--answer_file', type=str, required=True)
parser.add_argument('--predict_file', type=str, required=True)

parser.add_argument('--value', type=str, default="prognosis")

args = parser.parse_args()


actual = pd.read_csv(args.answer_file)
submission = pd.read_csv(args.predict_file)

actual = actual.sort_values(by=['id'], ignore_index=True)
submission = submission.sort_values(by=['id'], ignore_index=True)


def map_at_3(actual, predictions):
    """
    Calculate Mean Average Precision at 3 (MAP@3).

    每个样本最多 3 个预测（空格分隔），命中名次越靠前得分越高：1/rank。

    Parameters:
    actual (list): List of actual prognosis values.
    predictions (list of str): Space-separated up to 3 predicted prognosis values.

    Returns:
    float: The MAP@3 score.
    """
    total = len(actual)
    score = 0.0

    for act, preds in zip(actual, predictions):
        for rank, pred in enumerate(preds.split()[:3], start=1):
            if act == pred:
                score += 1.0 / rank
                break

    return score / total


# Compute MAP@3
performance = map_at_3(actual[args.value], submission[args.value])

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
