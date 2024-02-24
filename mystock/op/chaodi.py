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
from mystock.op.ma import MA, EMA, REF
from mystock.op.hhv import HHV, LLV, CROSS
import timeit


def calculate_chaodi(close):
    time1 = timeit.default_timer()
    # 计算移动平均线
    we5 = EMA(close, window=5)
    time2 = timeit.default_timer()
    x1 = we5 >= HHV(we5, 5)
    time3 = timeit.default_timer()
    x2 = we5 <= LLV(we5, 5)
    time4 = timeit.default_timer()

    flag1 = CROSS(x2, x1)
    ma59 = MA(close, 60)
    ma60 = np.concatenate((close[0:59], ma59))
    flag2 = close * 1.11 < ma60
    flag3 = ma60 < REF(ma60, 1)

    time5 = timeit.default_timer()

    # result = flag1 & flag2 & flag3
    chaodi = np.logical_and.reduce((flag1, flag2, flag3))
    # aa = result == dengtong
    time6 = timeit.default_timer()
    # print(f"time2-time1: {time2 - time1:.6f} seconds")
    # print(f"time3-time2: {time3 - time2:.6f} seconds")
    # print(f"time4-time3: {time4 - time3:.6f} seconds")
    # print(f"time5-time4: {time5 - time4:.6f} seconds")
    # print(f"time6-time5: {time6 - time5:.6f} seconds")
    return chaodi


if __name__ == '__main__':
    # 示例收盘价数据
    feature_open = np.array(
        [102, 103, 104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104, 100, 101, 102, 103,
         104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104, 100, 101, 102, 103, 104, 105,
         103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104, 100, 101])
    feature_close = np.array(
        [100, 101, 102, 103, 104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104, 100, 101,
         102, 103, 104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 104, 104, 100, 101, 102, 103,
         104, 105, 103, 107, 108, 108, 108, 107, 106, 105, 104, 105, 103, 104, 105, 99])

    feature_volume = np.array(
        [10000, 10100, 10200, 10300, 10400, 10500, 10300, 10700, 10800, 10800, 10800, 10700, 10600, 10500, 10400, 10005,
         10003, 10004, 10400, 10004, 10000, 10001,
         10002, 10300, 10004, 10005, 10003, 10007, 10008, 10008, 10800, 10007, 10006, 10005, 10400, 10005, 10003, 10004,
         10004, 10004, 10000, 10001, 10002, 10003,
         10004, 10500, 10300, 10007, 10008, 10008, 10008, 10007, 10006, 10500, 10004, 10005, 10003, 10004, 1000, 10004])

    # 计算邓通
    chaodi = calculate_chaodi(feature_close)
    print(chaodi)
