import pandas as pd

from helper.data_check_preparation import read_data
from helper.feature_engineering import feature_engineering
from helper.constant import PATH
from helper.preprocessing import standard_scaler


def train_model():
    # pembacaan dan pengecekan data
    df = read_data(PATH)

    # feature engineering
    df_transformed = feature_engineering(df)
    print("Start Saving Result Feature Engineering!")
    df_transformed.to_csv("artifacts/df_transformed.csv")


if __name__ == "__main__":
    print("START RUNNING PIPELINE!")
    train_model()
    print("FINISH RUNNING PIPELINE!")
