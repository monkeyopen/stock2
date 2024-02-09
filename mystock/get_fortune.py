#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
# 定义回测类
# 主要功能：
# 1，读入指定目录的数据，截取指定日期的数据
# 2，调用策略类，生成买卖信号
# 3，根据买卖信号和历史数据，计算最终受益

import datetime

from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np

from mystock.op.bias import calculate_bias
from mystock.op.rsi import calculate_rsi
from mystock.op.bollinger_bands import calculate_bollinger_bands
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

    def _set_signals(self, predictions, buy_signal=1, sell_signal=1):
        print(buy_signal, sell_signal)
        date = self.df['date'].values
        self.signals = np.zeros(len(self.df))
        idx = 0
        for i in range(0, len(self.df)):
            window_size = self.window_size + 60
            if i < window_size or date[i] < self.sample_start.strftime(self.dt_format) or date[
                i] > self.sample_end.strftime(
                self.dt_format):
                continue
            # 信号显著性
            if predictions[idx] > buy_signal:
                self.signals[i] = 1
            elif predictions[idx] < sell_signal:
                self.signals[i] = 2
            idx += 1
        self.df['signals'] = self.signals

    def _calculate_profit(self, stop_loss=1):
        """
        使用signals作为买卖信号，1是买，2是卖，其他为不操作。
        如果已经买入，就不再买了，如果已经卖出，可以再买
        todo 在指定的时间窗口上计算，小于数据的时间窗口。
        """
        position = 0
        self.profit = 1
        buy_price = 0

        for i in range(len(self.df) - 1):
            # 止损策略
            if stop_loss != 1 and position == 1 and self.df.iloc[i]['low'] < buy_price * stop_loss:
                print(stop_loss)
                position = 0
                # sell_price = self.df.iloc[i + 1]['open']
                sell_price = buy_price * stop_loss - 0.01
                profit = sell_price / buy_price
                self.profit *= profit
                print(
                    f"{self.df.iloc[i + 1]['date']}, 卖出价格 {sell_price}, 本次收益 {(profit - 1) * 100:.2f}%，总收益 {(self.profit - 1) * 100:.2f}%")
            elif self.df.iloc[i]['signals'] == 1 and position == 0:
                position = 1
                buy_price = self.df.iloc[i + 1]['open']
                print(f"{self.df.iloc[i + 1]['date']}, 买入价格 {buy_price}")
            elif self.df.iloc[i]['signals'] == 2 and position == 1:
                position = 0
                sell_price = self.df.iloc[i + 1]['open']
                profit = sell_price / buy_price
                self.profit *= profit
                print(
                    f"{self.df.iloc[i + 1]['date']}, 卖出价格 {sell_price}, 本次收益 {(profit - 1) * 100:.2f}%，总收益 {(self.profit - 1) * 100:.2f}%")
        if position == 1:
            sell_price = self.df.iloc[- 1]['close']
            profit = sell_price / buy_price
            self.profit *= profit
            print(
                f"{self.df.iloc[-1]['date']}, 卖出价格 {sell_price}, 本次收益 {(profit - 1) * 100:.2f}%，总收益 {(self.profit - 1) * 100:.2f}%")

    def add_strategy(self, strategy):
        self.strategy = strategy

    def moving_average(self, data, window_size):
        res = np.convolve(data, np.ones(window_size), 'valid') / window_size
        res = np.round(res, 3)
        return res

    def EMA(self, ndarry, window=7):
        weight = 2 / (window + 1)
        ema_values = np.zeros((len(ndarry),))
        # ema_values[:] = np.nan
        start_line = 0
        ema_values[start_line + window - 1] = ndarry[start_line:start_line + window - 1].mean()
        for i in range(start_line + window, len(ndarry)):
            ema_values[i] = (ndarry[i] * weight) + (ema_values[i - 1] * (1 - weight))
        return ema_values

    def generate_samples(self, label_type="buy"):
        features = []
        labels = []
        infos = []
        pres = []
        date = self.df['date'].values
        close_prices = self.df['close'].values
        volume = self.df['volume'].values
        feature_size = 0

        # 循环遍历生成样本数据，一次循环生成一条数据，所以循环的次数是样本窗口大小。单词循环中会读取特征需要的数据大小。
        for i in range(0, len(self.df)):
            window_size = self.window_size + 60
            if i < window_size or date[i] < self.sample_start.strftime(self.dt_format) or date[
                i] > self.sample_end.strftime(
                self.dt_format):
                continue

            # print(i, date[i])
            # 提取close价格作为特征
            # 定义一个将字符串日期转换为整数的函数
            def date_string_to_int(date_string):
                return int(date_string.replace("-", ""))

            # 使用NumPy的vectorize函数将函数应用于整个数组
            # date_string_to_int_vectorized = np.vectorize(date_string_to_int)
            # date_ints = date_string_to_int_vectorized(date[i - window_size:i + 1])
            # feature_date = date_ints % 100000
            feature_close = close_prices[i - window_size:i + 1]
            if i + 3 < len(self.df):
                target_close = close_prices[i + 3]
            else:
                target_close = 0
            # 处理负数价格
            min_close = np.min(feature_close)
            if min_close < 0:
                # 将所有价格加上最小负数价格的绝对值
                feature_close = feature_close - min_close + 1
                target_close = target_close - min_close + 1

            feature_close_normalized = feature_close / feature_close[-1]
            feature_close_normalized = np.clip(feature_close_normalized, a_min=0.1, a_max=10.0)
            target_close = target_close / feature_close[-1]
            # 计算MA5
            feature_ma5 = self.moving_average(feature_close_normalized, 5)
            # 计算MA10
            feature_ma10 = self.moving_average(feature_close_normalized, 10)
            # 计算MA20
            feature_ma20 = self.moving_average(feature_close_normalized, 20)
            # 计算MA30
            feature_ma30 = self.moving_average(feature_close_normalized, 30)
            # 计算MA60
            feature_ma60 = self.moving_average(feature_close_normalized, 60)
            # MA5>MA5
            feature_ma5_ma5 = np.array([(feature_ma5[-1] - feature_ma10[-2]) > 0])
            # MA5>MA10
            feature_ma5_ma10 = np.array([(feature_ma5[-1] - feature_ma10[-1]) > 0])
            # MA5>MA60
            feature_ma5_ma60 = np.array([(feature_ma5[-1] - feature_ma60[-1]) > 0])

            # 提取成交量作为特征
            feature_volume = volume[i - window_size:i + 1]
            feature_volume_normalized = feature_volume / feature_volume[-1]
            feature_volume_normalized = np.clip(feature_volume_normalized, a_min=0.5, a_max=2.0)
            # 计算MA5
            feature_volume_ma5 = self.moving_average(feature_volume_normalized, 5)
            # 计算MA10
            feature_volume_ma10 = self.moving_average(feature_volume_normalized, 10)

            # MACD
            ema5 = self.EMA(feature_close_normalized, window=5)
            ema12 = self.EMA(feature_close_normalized, window=12)
            ema26 = self.EMA(feature_close_normalized, window=26)
            dif = ema12 - ema26
            dea = self.EMA(dif, window=9)
            macd = np.array([dif[-2] <= dea[-2] and dif[-1] > dea[-1]])
            macd2 = np.array([dif[-2] >= dea[-2] and dif[-1] < dea[-1]])

            # RSI
            rsi14 = calculate_rsi(feature_close_normalized)
            rsi5 = calculate_rsi(feature_close_normalized, 5)
            rsi10 = calculate_rsi(feature_close_normalized, 10)

            # bollinger_bands
            middle_band, upper_band, lower_band = calculate_bollinger_bands(feature_close_normalized)
            # bias
            bias6 = calculate_bias(feature_close_normalized, window_size=6)
            bias12 = calculate_bias(feature_close_normalized, window_size=12)
            bias24 = calculate_bias(feature_close_normalized, window_size=24)
            bias5 = calculate_bias(feature_close_normalized, window_size=5)
            bias10 = calculate_bias(feature_close_normalized, window_size=10)
            bias20 = calculate_bias(feature_close_normalized, window_size=20)

            # 将所有特征添加到一个列表中
            feature_list = [
                # feature_date[-self.window_size:],
                # feature_close[-1:],  ##BRK.A 的收盘价太大，导致nan，先注释掉。
                feature_close_normalized[-self.window_size:],
                feature_ma5[-self.window_size:], feature_ma10[-self.window_size:],
                feature_ma20[-self.window_size:], feature_ma30[-self.window_size:],
                feature_ma60[-self.window_size:], feature_volume_normalized[-self.window_size:],
                feature_volume_ma5[-self.window_size:], feature_volume_ma10[-self.window_size:],
                ema5[-self.window_size:], middle_band[-1:], upper_band[-1:], lower_band[-1:], rsi14[-1:], rsi10[-1:],
                rsi5[-1:], bias6[-1:], bias12[-1:],
                bias24[-1:], bias5[-1:], bias10[-1:],
                bias20[-1:]]
            rounded_feature_list = [np.round(feature, decimals=3) for feature in feature_list]
            total_feature = rounded_feature_list + [feature_ma5_ma5, feature_ma5_ma10, feature_ma5_ma60, macd, macd2]
            # 使用 np.concatenate() 函数将特征列表连接起来
            feature = np.concatenate(total_feature)
            if np.any(np.isnan(feature)):
                print("Feature contains NaN values:", feature)

            if i == len(self.df) - 1:
                label = 0
                log_info = [self.df['date'].iloc[i], 0, 0, 0]
                pre = 0
            else:
                if label_type == "buy1":
                    # 1日内1个点涨幅
                    label = int(self.df['high'].iloc[i + 1] > self.df['close'].iloc[i] * 1.01)
                    # pre = int(feature_ma5_ma5 and feature_volume_normalized[-1] > feature_volume_normalized[-2] * 1.1)
                    pre = macd[-1]
                elif label_type == "buy5":
                    # 5日内5个点涨幅
                    label = int(max(self.df['high'].iloc[i + 1:i + 6]) > self.df['close'].iloc[i] * 1.05)
                    # pre = int(feature_ma5_ma5 and feature_volume_normalized[-1] > feature_volume_normalized[-2] * 1.1)
                    pre = macd[-1]
                elif label_type == "sell1":
                    label = int(self.df['low'].iloc[i + 1] < self.df['close'].iloc[i] * 0.99)
                    # pre = int(
                    #     not feature_ma5_ma5 and feature_volume_normalized[-1] > feature_volume_normalized[-2] * 1.1)
                    pre = macd2[-1]
                elif label_type == "sell5":
                    label = int(min(self.df['low'].iloc[i + 1:i + 6]) < self.df['close'].iloc[i] * 0.95)
                    # pre = int(
                    #     not feature_ma5_ma5 and feature_volume_normalized[-1] > feature_volume_normalized[-2] * 1.1)
                    pre = macd2[-1]
                elif label_type == "price":
                    label = target_close
                    pre = 1
                else:
                    label = 1
                    pre = 1

                # label = int(df['close'].iloc[i - 1] > df['close'].iloc[i - 2])
                # label = int(normalized_close_feature[-1] > normalized_close_feature[-2])

                log_info = [self.df['close'].iloc[i], self.df['high'].iloc[i + 1],
                            self.df['low'].iloc[i + 1]]
                log_info = [round(num, 2) for num in log_info]
                log_info = [self.df['date'].iloc[i]] + log_info
            features.append(feature)
            labels.append(label)
            infos.append(log_info)
            pres.append(pre)

        return features, labels, infos, pres

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
    stock_path = DATA_PATH + "/TSLA"
    backtest = Backtesting(
        data_dir=stock_path,
        dt_format='%Y-%m-%d',
        start_date=datetime.datetime(2023, 12, 1),
        end_date=datetime.datetime(2024, 1, 31),
        sample_start=datetime.datetime(2023, 12, 1),
        sample_end=datetime.datetime(2024, 1, 31)
    )

    backtest.add_strategy(Macd)

    backtest.run()
