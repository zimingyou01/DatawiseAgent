import os.path

import pandas as pd
import argparse

from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--answer_file', type=str, required=True)
parser.add_argument('--predict_file', type=str, required=True)

parser.add_argument('--value', type=str, default="open_channels")

args = parser.parse_args()


answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

answers = answers.sort_values('time')
predictions = predictions.sort_values('time')


y_true = answers[args.value].values
y_pred = predictions[args.value].values


performance = f1_score(y_true, y_pred, average='macro')


with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
