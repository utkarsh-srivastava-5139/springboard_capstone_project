"""Module for listing down additional custom functions required for production."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def log_transformer(data):
    return np.log(1 + data)


def scaling_transform(df):
    # Implement your scaling transformation here
    scaler = StandardScaler()
    cols = df.columns
    scaler.fit(df)
    df = scaler.transform(df)
    df = pd.DataFrame(data=df, columns=cols)
    return df


def mape(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted) / Y_actual)) * 100
    return mape
