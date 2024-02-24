import numpy as np


# 定义一个函数来计算DMA
def DMA(df, alpha):
    dma = np.zeros_like(df)
    dma[0] = df[0]

    for i in range(1, len(df)):
        dma[i] = (df[i] - dma[i - 1]) * alpha[i] + dma[i - 1]

    return dma


def MA(data, window_size):
    res = np.convolve(data, np.ones(window_size), 'valid') / window_size
    return res


def REF(series, n):
    return np.concatenate((np.full(n, np.nan), series[:-n]))


def EMA(ndarry, window=7):
    weight = 2 / (window + 1)
    ema_values = np.zeros((len(ndarry),))
    start_line = 0
    ema_values[start_line + window - 1] = ndarry[start_line:start_line + window - 1].mean()
    for i in range(start_line + window, len(ndarry)):
        ema_values[i] = (ndarry[i] * weight) + (ema_values[i - 1] * (1 - weight))
    return ema_values


if __name__ == '__main__':
    # 示例收盘价数据
    feature_close = np.array(
        [100, 101, 102, 103, 104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104])

    # 示例权重数据，这里我们使用一个0-1之间的随机数
    alpha = np.random.rand(len(feature_close))
    # 计算
    dma = DMA(feature_close, alpha)
    print(dma)
