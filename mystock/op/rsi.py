import pandas as pd


def calculate_rsi(data, window_length=14):
    # 计算涨跌幅
    delta = data.diff()
    delta = delta[1:]

    # 分别计算涨幅和跌幅的绝对值
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # 计算平均涨幅和平均跌幅
    roll_up = up.rolling(window_length).mean()
    roll_down = down.abs().rolling(window_length).mean()

    # 计算RS
    RS = roll_up / roll_down

    # 计算RSI
    RSI = 100.0 - (100.0 / (1.0 + RS))

    return RSI


if __name__ == '__main__':
    # 假设feature_close是一个pandas Series，保存了收盘价
    feature_close = pd.Series(...)

    # 计算14日RSI
    rsi = calculate_rsi(feature_close, window_length=14)
