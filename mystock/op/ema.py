import numpy as np
import pandas as pd

def EMA(df, window=7):
    weight = 2 / (window + 1)
    ema_values = np.empty((len(df),))
    ema_values[:] = np.nan
    start_line = df.first_valid_index()
    ema_values[start_line + window - 1] = df.loc[start_line:start_line + window - 1].mean()
    for i in range(start_line + window, len(df)):
        ema_values[i] = (df.iloc[i] * weight) + (ema_values[i - 1] * (1 - weight))
    return ema_values
    # return pd.DataFrame(ema_values, columns=["EMA"])