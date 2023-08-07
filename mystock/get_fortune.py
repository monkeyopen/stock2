#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
# 定义回测类
# 主要功能：
# 1，读入指定目录的数据，截取指定日期的数据
# 2，调用策略类，生成买卖信号
# 3，根据买卖信号和历史数据，计算最终受益

import datetime
import random

from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from strategy import Random
from macd import Macd
from mylog import log

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')


class Backtesting:
    def __init__(self, data_dir, dt_format, start_date, end_date):
        self.strategy = None
        self.data_dir = data_dir
        self.dt_format = dt_format
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        # Read data from the specified directory and extract data from the specified date
        self._read_data()

        # # Call the strategy class to generate buy and sell signals

        # aa = self.strategy(self.df)
        strategy = self.strategy(self.df)
        self.df = strategy.data
        # log(f"{self.df}")

        # Calculate the final profit based on the buy and sell signals and historical data
        self._calculate_profit()
        log(f"策略 {self.strategy.__module__}, 收益: {self.profit}")

    def _read_data(self):
        """
        从指定目录读取数据，提取指定日期的数据
        列定义为id,date,open,high,low,close,volume
        """
        df = pd.read_csv(self.data_dir)
        # 统一日期格式
        # print(self.dt_format)
        # df['date'] = pd.to_datetime(df['date'], format=self.dt_format)
        self.df = df.loc[(df['date'] >= self.start_date.strftime(self.dt_format)) & (
                df['date'] <= self.end_date.strftime(self.dt_format))]
        self.df = self.df.reset_index(drop=True)

    def _calculate_profit(self):
        """
        使用signals作为买卖信号，1是买，2是卖，其他为不操作。
        如果已经买入，就不再买了，如果已经卖出，可以再买
        todo 在指定的时间窗口上计算，小于数据的时间窗口。
        """
        position = 0
        self.profit = 0

        for i in range(len(self.df)):
            if self.df.iloc[i]['signals'] == 1 and position == 0:
                position = 1
                buy_price = self.df.iloc[i + 1]['open']
                log(f"{self.df.iloc[i+1]['date']}, 买入价格 {buy_price}")
            elif self.df.iloc[i]['signals'] == 2 and position == 1:
                position = 0
                sell_price = self.df.iloc[i + 1]['open']
                self.profit += sell_price - buy_price
                log(f"{self.df.iloc[i + 1]['date']}, 卖出价格 {sell_price}")
        if position == 1:
            self.profit += self.df.iloc[-1]['close'] - buy_price

    def add_strategy(self, strategy):
        self.strategy = strategy

    # def _generate_signals(self):
    #     """
    #     生成和self.df等长的一个list，全是0，随机选2段，填成1
    #     """
    #     self.signals = np.zeros(len(self.df))
    #     idx_list = random.sample(range(len(self.df)), 4)
    #     idx_list = sorted(idx_list)
    #     print(idx_list)
    #     self.signals[idx_list[0]:idx_list[1]] = 1
    #     self.signals[idx_list[2]:idx_list[3]] = 1
    #     self.df["signals"] = self.signals


if __name__ == '__main__':
    # 00700
    # 09888
    stock_path = DATA_PATH + "09888"
    backtest = Backtesting(
        data_dir=stock_path,
        dt_format='%Y-%m-%d',
        start_date=datetime.datetime(2023, 1, 1),
        end_date=datetime.datetime(2023, 12, 31)
    )

    backtest.add_strategy(Macd)

    backtest.run()
