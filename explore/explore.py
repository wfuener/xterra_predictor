import json

import mlflow
import pandas as pd
import time
from datetime import datetime
import numpy as np
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from ydata_profiling import ProfileReport

BASE_DIR = os.path.join(os.path.dirname(__file__))
INPUT_FILE = f"{BASE_DIR}/age_group_full.json"
mlflow.set_tracking_uri("sqlite:///mlruns.db")


def main():
    df = ingest()
    explore_data(df)
    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file("artifacts/profile.html")


def ingest():
    with open(INPUT_FILE, 'r') as file:
        data = json.loads(file.read())

    for row in data:
        row['place'] = str_to_int(row['place'])
        row['bib'] = str_to_int(row['bib'])
        row['age'] = str_to_int(row['age'])
        row['swim_time'] = convert_elapsed_time(row['swim_time'])
        row['t1_time'] = convert_elapsed_time(row['t1_time'])
        row['bike_time'] = convert_elapsed_time(row['bike_time'])
        row['t2_time'] = convert_elapsed_time(row['t2_time'])
        row['run_time'] = convert_elapsed_time(row['run_time'])
        row['chip_time'] = convert_elapsed_time(row['chip_time'])

    return pd.DataFrame(data)


def str_to_int(_str):
    try:
        integer_value = int(_str)
        return integer_value
    except ValueError:
        return None


def convert_elapsed_time(elapsed_str) -> datetime:
    def convert(string, format):
        try:
            struct_time = time.strptime(string, format)
        except ValueError:
            print(f"bad value found while parsing elasped time: {string}")
            return None
        # convert to seconds
        total_seconds = (
                struct_time.tm_hour * 3600 +
                struct_time.tm_min * 60 +
                struct_time.tm_sec
        )
        return total_seconds

    # first check for microseconds, if not add them as 0. Some values
    # have them so it's easier to append microseconds to all for conversion
    if len(elapsed_str.split(".")) == 1:
        elapsed_str = f"{elapsed_str}.0"
    # convert
    if len(elapsed_str.split(":")) == 3:
        return convert(elapsed_str, "%H:%M:%S.%f")
    elif len(elapsed_str.split(":")) == 2:
        return convert(elapsed_str, "%M:%S.%f")
    else:
        print(f"bad value found while parsing elasped time: {elapsed_str}")
        return None


def explore_data(df: pd.DataFrame):
    # Write csv from stats dataframe
    description = df.describe()
    description.index.name = 'label' # index name would be otherwise be null when writing to csv.
    description.to_csv('artifacts/dataset_statistics.csv')
    # Log CSV to MLflow
    # mlflow.log_artifact('dataset_statistics.csv')

    # run correlations on all numerical values
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_cols = df.select_dtypes(include=numerics)

    # Compute the correlation matrix
    corr = numerical_cols.corr()
    corr.to_csv("artifacts/correlations.csv")
    # mlflow.log_artifact('correlations.csv')

    # Draw the heatmap with the mask and correct aspect ratio
    heatmap = sns.heatmap(corr, annot=True, cmap="crest")
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    heatmap.get_figure().savefig("artifacts/correlation_heatmap.png")
    # plt.show()
    # mlflow.log_figure(heatmap.get_figure(), "correlation_heatmap.png")
    return df


if __name__ == '__main__':
    main()