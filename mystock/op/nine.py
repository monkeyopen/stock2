import numpy as np


def calculate_nine(close, window_length=4, flag="up"):
    if flag == "up":
        a1 = close[window_length:] > close[:-window_length]
    else:
        a1 = close[window_length:] < close[:-window_length]
    a2 = np.concatenate((np.full(window_length, False), a1))
    zero_array = np.zeros_like(close)
    for i in range(1, len(close)):
        if a2[i]:
            zero_array[i] = zero_array[i - 1] + 1
        else:
            zero_array[i] = 0

    return zero_array


if __name__ == '__main__':
    # 示例收盘价数据
    feature_close = np.array(
        [100, 101, 102, 103, 104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104])

    # 计算神奇九转
    nine = calculate_nine(feature_close, window_length=4)
    print(nine)
