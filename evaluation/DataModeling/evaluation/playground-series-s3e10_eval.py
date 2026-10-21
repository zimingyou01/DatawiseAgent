import os.path

import pandas as pd
import argparse
from sklearn.metrics import log_loss


parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--answer_file', type=str, required=True)
parser.add_argument('--predict_file', type=str, required=True)

parser.add_argument('--value', type=str, default="Class")

args = parser.parse_args()

answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

answers = answers.sort_values(by=['id'], ignore_index=True)
predictions = predictions.sort_values(by=['id'], ignore_index=True)

# Compute log loss
performance = log_loss(answers[args.value], predictions[args.value])

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
