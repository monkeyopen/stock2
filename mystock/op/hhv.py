import numpy as np


def HHV(series, n):
    return np.array([np.max(series[i - n + 1:i + 1]) if i >= n - 1 else series[i] for i in range(len(series))])


def LLV(series, n):
    return np.array([np.min(series[i - n + 1:i + 1]) if i >= n - 1 else series[i] for i in range(len(series))])


def CROSS(series1, series2):
    cross = (series1[:-1] < series2[:-1]) & (series1[1:] > series2[1:])
    result = np.concatenate(([False], cross))
    return result


if __name__ == '__main__':
    # 示例收盘价数据
    feature_close = np.array(
        [100, 101, 102, 103, 104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104])

    # 计算HHV
    hhv = HHV(feature_close, 5)
    print(hhv)

    # 计算LLV
    llv = LLV(feature_close, 5)
    print(llv)

    # 示例数据
    x1 = np.random.rand(100)
    x2 = np.random.rand(100)

    # 计算CROSS
    cross = CROSS(x2, x1)
    print(cross)
