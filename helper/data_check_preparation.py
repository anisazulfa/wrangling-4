import pandas as pd
import numpy as np


def read_data(PATH):
    """
    Read data from dataset from path

    Parameters
    ----------
    PATH : str
        path source of training data, csv.

    Returns
    -------
    data : pd.DataFrame
        Data for modeling
    """
    data = pd.read_csv(PATH, encoding="latin-1")

    return data
