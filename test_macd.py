# 要回测MACD策略的收益水平，首先需要获取腾讯科技的历史数据。这里假设您已经有了一个函数可以获取这些数据。接下来，我们将实现一个简单的MACD策略回测。
#
# 计算MACD指标：MACD指标包括DIF（快线）、DEA（慢线）和MACD柱。计算方法如下：
#
# DIF = 12日EMA - 26日EMA
# DEA = 9日EMA（DIF）
# MACD柱 = 2 * (DIF - DEA)
# 设定交易信号：当DIF上穿DEA时，买入；当DIF下穿DEA时，卖出。
#
# 计算策略收益：根据交易信号，计算策略的累计收益。
#
# 以下是一个简单的Python代码示例：


import pandas as pd
import numpy as np
from get_data import read_stock_data


# 假设您已经有了一个函数可以获取腾讯科技的历史数据
# 为了成功执行macd(tencent_data)函数，tencent_data应该是一个包含股票历史价格数据的Pandas DataFrame。这个DataFrame应该至少包含以下三列：日期（通常是索引）、收盘价（Close）和交易量（Volume）。MACD指标主要关注收盘价，所以至少需要包含这一列。
#
# 以下是一个简单的tencent_data DataFrame示例：
#
# 复制
#             Date        Close       Volume
# 0    2023-01-01    100.00    1500000
# 1    2023-01-02    101.50    2000000
# 2    2023-01-03    102.00    1800000
# 3    2023-01-04    103.50    2100000
# ...
# 在这个示例中，日期是索引，收盘价和交易量分别是Close和Volume列。当你将这样的数据传递给macd(tencent_data)函数时，它应该能够正确计算MACD指标


def get_tencent_data(start_date, end_date):
    df = read_stock_data("data/00700")
    df['Date'] = df['date']
    df = df.set_index("Date")
    result = df.loc[start_date:end_date]
    return result


# 计算EMA
def ema(data, n):
    return data.ewm(span=n, adjust=False).mean()


# 计算MACD指标
def macd(data):
    data['DIF'] = ema(data['close'], 12) - ema(data['close'], 26)
    data['DEA'] = ema(data['DIF'], 9)
    data['MACD'] = 2 * (data['DIF'] - data['DEA'])
    return data


# 回测MACD策略
def backtest_macd(data):
    data['Signal'] = np.where(data['DIF'] > data['DEA'], 1, -1)
    data['Return'] = data['close'].pct_change() * data['Signal'].shift(1)
    data['Cumulative_Return'] = (1 + data['Return']).cumprod()
    return data


if __name__ == '__main__':
    # 获取腾讯科技的历史数据
    start_date = '2023-01-01'
    end_date = '2023-04-15'
    tencent_data = get_tencent_data(start_date, end_date)

    # 计算MACD指标
    tencent_data = macd(tencent_data)

    # 回测MACD策略
    tencent_data = backtest_macd(tencent_data)

    # 输出策略收益
    print("MACD策略的累计收益：", tencent_data['Cumulative_Return'].iloc[-1])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(tencent_data[['date', 'close', 'DIF', 'DEA', 'Signal', 'MACD', 'Return']])
    print(len(tencent_data))
    print(tencent_data.loc["2023-01-18", "high"])
    print(tencent_data.iloc[4, 3])
    print(tencent_data["open"][2])
