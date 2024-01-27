import pandas as pd


def calculate_bollinger_bands(data, window_length=20, num_std=2):
    # 计算移动平均线
    moving_avg = data.rolling(window=window_length).mean()

    # 计算标准差
    std_dev = data.rolling(window=window_length).std()

    # 计算上轨和下轨
    upper_band = moving_avg + (std_dev * num_std)
    lower_band = moving_avg - (std_dev * num_std)

    return moving_avg, upper_band, lower_band


if __name__ == '__main__':
    # 假设feature_close是一个pandas Series，保存了收盘价
    feature_close = pd.Series(...)

    # 计算布林带
    middle_band, upper_band, lower_band = calculate_bollinger_bands(feature_close)
