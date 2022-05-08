import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import normalize


def get_dummies(df: pd.DataFrame, categoric_cols: list):
    output_df = pd.get_dummies(df, columns=categoric_cols, dummy_na=False)
    return output_df


def standard_scaler(df: pd.DataFrame):
    """Scaling standard scaler transform."""
    index_cols = df.index
    scaler = preprocessing.Normalizer()
    np_scaler = scaler.fit_transform(df)
    df_transformed = pd.DataFrame(np_scaler, index=index_cols, columns=df.columns)
    return scaler, df_transformed
