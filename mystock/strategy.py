#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
# 定义策略基类
# 主要功能：
# 1，初始化
# 2，定义next函数，生成买卖信号，
import random

import numpy as np


class Strategy:
    def __init__(self, data):
        self.data = data

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        # dt = dt or self.data['date']
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        pass


class Random(Strategy):
    def __init__(self, data):
        super().__init__(data)
        self.random()

    def random(self):
        """
        生成和self.df等长的一个list，全是0，随机选2段，填成1
        """
        self.signals = np.zeros(len(self.data))
        idx_list = random.sample(range(len(self.data)), 4)
        idx_list = sorted(idx_list)
        print(idx_list)
        self.signals[idx_list[0]:idx_list[1]] = 1
        self.signals[idx_list[2]:idx_list[3]] = 1
        self.data["signals"] = self.signals
        print(self.signals)


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
