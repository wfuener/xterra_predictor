import pandas as pd
from sklearn.model_selection import train_test_split


def split(df: pd.DataFrame):
    train, test = train_test_split(df, test_size=0.2)
    return train, test
