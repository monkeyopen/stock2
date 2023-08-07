# -*- coding:utf-8 -*-
# Python 实用宝典
# 量化投资原来这么简单(1)
# 2020/04/12

import backtrader as bt
import datetime
from TestStrategy import TestStrategy
from SmaCross import SmaCross
from Macd import Macd
from dotenv import load_dotenv
import os

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')


if __name__ == '__main__':
    # 初始化模型
    cerebro = bt.Cerebro()
    # 构建策略
    # strats = cerebro.addstrategy(bt.indicators.MACD)
    strats = cerebro.addstrategy(Macd)

    # data = bt.feeds.YahooFinanceCSVData(
    #     dataname='数据文件所在位置',
    #     fromdate=datetime.datetime(2000, 1, 1),
    #     todate=datetime.datetime(2000, 12, 31)
    # )
    # 需要设定每个列的含义，比如开盘价在第4列，则open = 3（从0开始算起），如下所示:
    # data = bt.feeds.YahooFinanceCSVData(dataname=DATA_PATH)
    data = bt.feeds.GenericCSVData(
        dataname=DATA_PATH,
        datetime=1,
        open=2,
        high=3,
        low=4,
        close=5,
        volume=6,
        dtformat=('%Y-%m-%d'),
        fromdate=datetime.datetime(2023, 1, 1),
        todate=datetime.datetime(2023, 5, 22)
    )

    cerebro.adddata(data)
    # 设定初始资金
    cerebro.broker.setcash(100000.0)
    # 现实生活中的股票交易里，每次交易都需要支付一定的佣金，比如万五（交易额每满一万元收取5元佣金）万三等，在Backtrader里你只需要这么设定即可：
    cerebro.broker.setcommission(0.000)
    # 设定需要设定每次交易买入的股数，可以这样
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)

    # 策略执行前的资金
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.run()

    # 策略执行后的资金
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()
