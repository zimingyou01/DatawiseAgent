import os.path

import pandas as pd
import argparse
from math import sqrt
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--answer_file', type=str, required=True)
parser.add_argument('--predict_file', type=str, required=True)

parser.add_argument('--value', type=str, default="emission")

args = parser.parse_args()

answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

answers = answers.sort_values(by=['ID_LAT_LON_YEAR_WEEK'], ignore_index=True)
predictions = predictions.sort_values(by=['ID_LAT_LON_YEAR_WEEK'], ignore_index=True)

# 提取预测值和实际标签
predicted_values = predictions[args.value].values
actual_values = answers[args.value].values

# Compute RMSE
rmse = sqrt(mean_squared_error(actual_values, predicted_values))

performance = rmse

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
