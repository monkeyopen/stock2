import numpy as np


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def calculate_rsi(data, window_length=14):
    # 计算涨跌幅
    delta = np.diff(data)

    # 分别计算涨幅和跌幅的绝对值
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # 计算平均涨幅和平均跌幅
    roll_up = np.mean(rolling_window(up, window_length), axis=-1)
    roll_down = np.mean(rolling_window(np.abs(down), window_length), axis=-1)

    # 计算RS
    RS = np.full_like(roll_up, np.inf)
    index = roll_down != 0
    RS[index] = roll_up[index] / roll_down[index]

    # 计算RSI
    RSI = (100.0 - (100.0 / (1.0 + RS))) / 100.0

    return RSI


if __name__ == '__main__':
    # 示例收盘价数据
    feature_close = np.array(
        [100, 101, 102, 103, 104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104])

    # 计算14日RSI
    rsi = calculate_rsi(feature_close, window_length=14)
    print(rsi)
