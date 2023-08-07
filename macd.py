# 导入必要的库
import pandas as pd
import mplfinance as mpf
from stockstats import StockDataFrame as Sdf



if __name__ == '__main__':
    # 读取00700的股票数据
    df = pd.read_csv('data/00700')

    # 将日期列转换为datetime类型
    df['date'] = pd.to_datetime(df['date'])

    # 设置日期列为索引
    df.set_index('date', inplace=True)

    # 筛选出2022年元旦至2023年4月18日的数据
    df = df.loc['2023-01-01':'2023-04-18']

    # 将数据转换为StockDataFrame类型
    stock = Sdf.retype(df)

    # 计算MACD指标
    stock['macd']

    # 绘制K线图和MACD指标
    mpf.plot(stock, type='candle', mav=(5, 10, 20), volume=True, figratio=(16,9), style='charles', title='00700股票K线图和MACD指标')