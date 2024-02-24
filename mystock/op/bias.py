import numpy as np


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def calculate_bias(data, window_size=6):
    # 计算移动平均线
    moving_avg = np.mean(rolling_window(data, window_size), axis=-1)

    # 计算BIAS
    bias = np.full_like(moving_avg, np.nan)
    index = moving_avg != 0
    bias[index] = (data[window_size - 1:][index] - moving_avg[index]) / moving_avg[index] * 10

    return bias


if __name__ == '__main__':
    # 示例收盘价数据
    feature_close = np.array(
        [100, 101, 102, 103, 104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104])

    # 计算6日BIAS
    bias = calculate_bias(feature_close, window_size=6)
    print(bias)
