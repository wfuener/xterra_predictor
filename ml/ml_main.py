import os
import mlflow
from steps.ingest import ingest
from steps.split import split
from steps.train import train_model
from steps.evaulate import test_model

mlflow.set_tracking_uri("sqlite:///mlruns.db")

BASE_DIR = os.path.join(os.path.dirname(__file__))
INPUT_FILE = f"{BASE_DIR}/age_group_full.json"


def main():
  with mlflow.start_run() as active_run:
    print("Launching process")
    df = ingest(INPUT_FILE)
    train, test = split(df)
    regression = train_model(train)
    test_model(test, regression)


if __name__ == '__main__':
    main()

