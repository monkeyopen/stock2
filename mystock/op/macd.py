import pandas as pd


def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    # 计算短期EMA和长期EMA
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()

    # 计算DIF
    DIF = short_ema - long_ema

    # 计算DEA
    DEA = DIF.ewm(span=signal_window, adjust=False).mean()

    # 计算MACD柱状图
    MACD_hist = DIF - DEA

    return DIF, DEA, MACD_hist


if __name__ == '__main__':
    # 假设feature_close是一个pandas Series，保存了收盘价
    feature_close = pd.Series(...)

    # 计算MACD
    dif, dea, macd_hist = calculate_macd(feature_close)
