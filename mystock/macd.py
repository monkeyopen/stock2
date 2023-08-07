#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
# macd策略类
# 主要功能：
# 1，初始化
# 2，生成买卖信号，

import random
import numpy as np
from strategy import Strategy
from op.ema import EMA

class Macd(Strategy):
    def __init__(self, data):
        super().__init__(data)
        # self.dataclose = self.data.close
        # self.volume = self.data.volume


        self.data['ema12'] = EMA(self.data['close'], window=12)
        self.data['ema26'] = EMA(self.data['close'], window=26)
        self.data['dif'] = self.data['ema12'] - self.data['ema26']
        self.data['dea'] = EMA(self.data['dif'], window=9)

        self.signals = np.zeros(len(self.data))
        start_line = self.data['dea'].first_valid_index()
        for i in range(start_line + 1, len(self.data)):
            condition1 = self.data['dif'].iloc[i-1] - self.data['dea'].iloc[i-1]
            condition2 = self.data['dif'].iloc[i] - self.data['dea'].iloc[i]
            if condition1 <= 0 and condition2 > 0:
                self.signals[i] = 1
            if condition1 >= 0 and condition2 < 0:
                self.signals[i] = 2

        self.data['signals'] = self.signals


    # def next(self):
    #     if not self.position:
    #         condition1 = self.macd[-1] - self.signal[-1]
    #         condition2 = self.macd[0] - self.signal[0]
    #         if condition1 < 0 and condition2 > 0:
    #             self.log('BUY CREATE, %.2f' % self.dataclose[0])
    #             self.order = self.buy()
    #
    #     else:
    #         condition1 = self.macd[-1] - self.signal[-1]
    #         condition2 = self.macd[0] - self.signal[0]
    #         if condition1 >0 and condition2 < 0:
    #         #     self.log('BUY CREATE, %.2f' % self.dataclose[0])
    #         #     self.order = self.buy()
    #         # condition = (self.dataclose[0] - self.bar_executed_close) / self.dataclose[0]
    #         # if condition > 0.05 or condition < -0.1:
    #             self.log('SELL CREATE, %.2f' % self.dataclose[0])
    #             self.order = self.sell()

