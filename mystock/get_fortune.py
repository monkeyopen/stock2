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
    def __init__(self, data_dir, dt_format, start_date, end_date, sample_start, sample_end):
        self.strategy = None
        self.data_dir = data_dir
        self.dt_format = dt_format
        self.start_date = start_date
        self.end_date = end_date
        self.sample_start = sample_start
        self.sample_end = sample_end
        self.window_size = 10

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
        df = df[(df['volume'] != 0) & (df['close'] != 0) & (df['open'] != 0)]
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
                log(f"{self.df.iloc[i + 1]['date']}, 买入价格 {buy_price}")
            elif self.df.iloc[i]['signals'] == 2 and position == 1:
                position = 0
                sell_price = self.df.iloc[i + 1]['open']
                self.profit += sell_price - buy_price
                log(f"{self.df.iloc[i + 1]['date']}, 卖出价格 {sell_price}")
        if position == 1:
            self.profit += self.df.iloc[-1]['close'] - buy_price

    def add_strategy(self, strategy):
        self.strategy = strategy

    def moving_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size), 'valid') / window_size

    def generate_samples(self, buy=1):
        features = []
        labels = []
        infos = []
        date = self.df['date'].values
        close_prices = self.df['close'].values
        volume = self.df['volume'].values
        feature_size = 0

        # 循环遍历生成样本数据，一次循环生成一条数据，所以循环的次数是样本窗口大小。单词循环中会读取特征需要的数据大小。
        for i in range(0, len(self.df)):
            if date[i] < self.sample_start.strftime(self.dt_format) or date[i] > self.sample_end.strftime(
                    self.dt_format):
                continue
            # print(i, date[i])
            # 提取close价格作为特征
            window_size = self.window_size + 60
            feature_close = close_prices[i - window_size:i + 1]
            feature_close_normalized = feature_close / feature_close[-1]

            # 提取成交量作为特征
            feature_volume = volume[i - window_size:i + 1]
            feature_volume_normalized = feature_volume / feature_volume[-1]

            # 计算MA5
            feature_ma5 = self.moving_average(feature_close_normalized, 5)
            # 计算MA10
            feature_ma10 = self.moving_average(feature_close_normalized, 10)
            # 计算MA20
            feature_ma20 = self.moving_average(feature_close_normalized, 20)
            # 计算MA30
            feature_ma30 = self.moving_average(feature_close_normalized, 30)
            # 计算MA30
            feature_ma40 = self.moving_average(feature_close_normalized, 40)
            # 计算MA30
            feature_ma50 = self.moving_average(feature_close_normalized, 50)
            # 计算MA30
            feature_ma60 = self.moving_average(feature_close_normalized, 60)

            # 将所有特征添加到一个列表中
            feature_list = [feature_close_normalized[-self.window_size:], feature_volume_normalized[-self.window_size:],
                            feature_ma5[-self.window_size:], feature_ma10[-self.window_size:],
                            feature_ma20[-self.window_size:], feature_ma30[-self.window_size:],
                            feature_ma40[-self.window_size:], feature_ma50[-self.window_size:],
                            feature_ma60[-self.window_size:]]

            # 使用 np.concatenate() 函数将特征列表连接起来
            feature = np.concatenate(feature_list)

            feature_size = feature.size
            # print(df['close'].iloc[i + 1], df['open'].iloc[i + 1])
            if i == len(self.df) - 1:
                label = 0
                log_info = [self.df['date'].iloc[i], 0, 0]
            else:
                if buy == 1:
                    label = int(self.df['close'].iloc[i + 1] > self.df['open'].iloc[i + 1] * 1.01)
                else:
                    label = int(self.df['close'].iloc[i + 1] < self.df['open'].iloc[i + 1] * 0.99)
                # label = int(df['close'].iloc[i - 1] > df['close'].iloc[i - 2])
                # label = int(normalized_close_feature[-1] > normalized_close_feature[-2])

                log_info = [self.df['date'].iloc[i], self.df['open'].iloc[i + 1], self.df['close'].iloc[i + 1]]

            features.append(feature)
            labels.append(label)
            infos.append(log_info)

        return features, labels, feature_size, infos

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
    stock_path = DATA_PATH + "/09888"
    backtest = Backtesting(
        data_dir=stock_path,
        dt_format='%Y-%m-%d',
        start_date=datetime.datetime(2023, 1, 1),
        end_date=datetime.datetime(2023, 12, 31)
    )

    backtest.add_strategy(Macd)

    backtest.run()
