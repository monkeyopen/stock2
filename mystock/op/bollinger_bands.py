import numpy as np


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def calculate_bollinger_bands(data, window_length=20, num_std=2):
    # 计算移动平均线
    moving_avg = np.mean(rolling_window(data, window_length), axis=-1)

    # 计算标准差
    std_dev = np.std(rolling_window(data, window_length), axis=-1)

    # 计算上轨和下轨
    upper_band = moving_avg + (std_dev * num_std)
    lower_band = moving_avg - (std_dev * num_std)

    return moving_avg, upper_band, lower_band


if __name__ == '__main__':
    # 示例收盘价数据
    feature_close = np.array(
        [100, 101, 102, 103, 104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104])

    # 计算布林带
    middle_band, upper_band, lower_band = calculate_bollinger_bands(feature_close)
