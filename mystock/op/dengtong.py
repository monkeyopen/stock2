from dotenv import load_dotenv
import os

load_dotenv()
CONF_PATH = os.getenv('CONF_PATH')
DATA_PATH = os.getenv('DATA_PATH')
ROOT_PATH = os.getenv('ROOT_PATH')
import sys

# print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([ROOT_PATH])

import numpy as np
from mystock.op.ma import DMA, MA
import timeit


def calculate_dengtong(open, close, volume):
    time1 = timeit.default_timer()
    # 计算移动平均线
    ma4 = MA(volume, 4)
    time12 = timeit.default_timer()
    # 为了使权重数组和收盘价数组长度一致，我们在ma_vol_4前面补3个volume
    ma4 = np.concatenate((np.array(volume[:3]), ma4))

    time13 = timeit.default_timer()

    # 计算权重：vol / ma_vol_4 / 4
    weights = volume / ma4 / 4
    time14 = timeit.default_timer()
    # 计算VAR31A
    var31a = DMA(close, weights)

    time15 = timeit.default_timer()

    # 计算 (CLOSE - VAR31A) / VAR31A * 100
    diff_percentage = (close - var31a) / var31a * 100

    time2 = timeit.default_timer()

    ma32 = MA(volume, 32)
    ma32 = np.concatenate((np.array(volume[:31]), ma32))
    # 计算权重：vol / ma_vol_4 / 4
    weights32 = volume / ma32 / 32
    # 计算VAR31A
    var32a = DMA(close, weights32)

    # 判断结果是否小于-8
    var31 = diff_percentage < -8

    var32 = (var31a - var32a) / var32a * 100 < -21

    time3 = timeit.default_timer()

    # 计算REF(CLOSE, 1)，即将收盘价数组向后移动一位
    ref_close_1 = np.concatenate((close[0:1], close[:-1]))
    # 计算 (OPEN - REF(CLOSE, 1)) / REF(CLOSE, 1)
    diff_percentage = (open - ref_close_1) / ref_close_1
    # 判断结果是否大于-0.05
    var33 = diff_percentage > -0.05

    time4 = timeit.default_timer()

    # VAR34 := COUNT(NOT(C=O), 8) >= 8;
    # 计算收盘价是否不等于开盘价
    not_equal = close != open

    # 使用convolve函数计算过去8天内收盘价不等于开盘价的天数
    not_equal_count = np.convolve(not_equal.astype(int), np.ones(8, dtype=int), 'same')

    # 判断这个天数是否大于等于8
    var34 = not_equal_count >= 8

    time5 = timeit.default_timer()

    dengtong = np.logical_and.reduce((var31, var32, var33, var34))

    time6 = timeit.default_timer()
    # print(
    #     f"time2-time1: {time2 - time1:.6f} seconds, time12-time1: {time12 - time1:.6f} seconds,time13-time12: {time13 - time12:.6f} seconds,time14-time13: {time14 - time13:.6f} seconds,time15-time14: {time15 - time14:.6f} seconds,time2-time15: {time2 - time15:.6f} seconds")
    # print(f"time3-time2: {time3 - time2:.6f} seconds")
    # print(f"time4-time3: {time4 - time3:.6f} seconds")
    # print(f"time5-time4: {time5 - time4:.6f} seconds")
    # print(f"time6-time5: {time6 - time5:.6f} seconds")
    return dengtong


if __name__ == '__main__':
    # 示例收盘价数据
    feature_open = np.array(
        [102, 103, 104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104, 100, 101, 102, 103,
         104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104, 100, 101, 102, 103, 104, 105,
         103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104, 100, 101])
    feature_close = np.array(
        [100, 101, 102, 103, 104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104, 100, 101,
         102, 103, 104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104, 100, 101, 102, 103,
         104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104])

    feature_volume = np.array(
        [10000, 10100, 10200, 10300, 10400, 10500, 10300, 10700, 10800, 10800, 10800, 10700, 10600, 10500, 10400, 10005,
         10003, 10004, 10400, 10004, 10000, 10001,
         10002, 10300, 10004, 10005, 10003, 10007, 10008, 10008, 10800, 10007, 10006, 10005, 10400, 10005, 10003, 10004,
         10004, 10004, 10000, 10001, 10002, 10003,
         10004, 10500, 10300, 10007, 10008, 10008, 10008, 10007, 10006, 10500, 10004, 10005, 10003, 10004, 1000, 10004])

    # 计算邓通
    dengtong = calculate_dengtong(feature_open, feature_close, feature_volume)
    # print(dengtong)
