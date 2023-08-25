"""
This module defines the following routines used by the 'ingest' step of the regression recipe:

- ``load_file_as_dataframe``: Defines customizable logic for parsing dataset formats that are not
  natively parsed by MLflow Recipes (i.e. formats other than Parquet, Delta, and Spark SQL).
"""
from pandas import DataFrame

import json

import mlflow
import pandas as pd
import time
from datetime import datetime
import numpy as np
import os


def ingest(location: str) -> DataFrame:
    with open(location, 'r') as file:
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

    print("ingestion finished")
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


