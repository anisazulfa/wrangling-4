import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from helper.preprocessing import get_dummies, standard_scaler


def feature_engineering(df):
    # 2
    dict_col = {
        "dateCreated": "ad_created",
        "dateCrawled": "date_crawled",
        "fuelType": "fuel_type",
        "lastSeen": "last_seen",
        "monthOfRegistration": "registration_month",
        "notRepairedDamage": "unrepaired_damage",
        "nrOfPictures": "num_of_pictures",
        "offerType": "offer_type",
        "postalCode": "postal_code",
        "powerPS": "power_ps",
        "vehicleType": "vehicle_type",
        "yearOfRegistration": "registration_year",
    }
    df.rename(columns=dict_col, inplace=True)
    # 3
    df[["ad_created", "date_crawled", "last_seen"]] = df[
        ["ad_created", "date_crawled", "last_seen"]
    ].apply(pd.to_datetime)
    # 4
    df["price"] = df["price"].str.replace("$", "")
    df["price"] = df["price"].str.replace(",", "")
    df["price"] = df["price"].astype("float")
    df["price"] = df["price"].astype("int")
    df["odometer"] = df["odometer"].str.replace("km", "")
    df["odometer"] = df["odometer"].str.replace(",", "")
    df["odometer"] = df["odometer"].astype("float")
    df["odometer"] = df["odometer"].astype("int")
    # 5
    col_drop = []
    for col in df:
        if df[col].dtypes == int and (
            df[col].nunique() < 2 or df[col].nunique() > 5000
        ):
            col_drop.append(col)
        if df[col].dtypes == object:
            if df[col].nunique() > 10:
                col_drop.append(col)
    df.drop(col_drop, axis=1, inplace=True)
    # 6
    df = df[(df["price"] >= 500) & (df["price"] <= 40000)]
    # 7
    for col in df:
        if df[col].dtypes == object:
            df[col] = df[col].apply(lambda x: df[col].mode()[0] if pd.isna(x) else x)
        if df[col].dtypes == int:
            df[col] = df[col].apply(lambda x: df[col].median() if np.isnan(x) else x)
    # 8
    numeric_cols = [
        col
        for col in df.columns
        if (df[col].dtype == "float" or df[col].dtype == "int") and col != "price"
    ]
    df_numeric = df[numeric_cols]
    scaler, df_transformed = standard_scaler(df_numeric)
    for col in numeric_cols:
        df[col] = df_transformed[col]
    # 9
    categoric_cols = [col for col in df.columns if (df[col].dtype == "object")]
    output_df = get_dummies(df, categoric_cols)

    return output_df
