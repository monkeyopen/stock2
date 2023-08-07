# 导入需要的库
import backtrader as bt
import akshare as ak
from get_data import read_stock_data

# 定义策略
class MyStrategy(bt.Strategy):
    def __init__(self):
        self.macd = bt.indicators.MACD(self.data.close)

    def next(self):
        if not self.position:
            if self.macd.macd[0] > self.macd.signal[0]:
                self.buy()
        elif self.macd.macd[0] < self.macd.signal[0]:
            self.close()


if __name__ == '__main__':
    # 获取腾讯科技历史数据
    data = bt.feeds.PandasData(dataname=ak.stock_zh_a_hist(symbol='600519'))
    # data = read_stock_data("data/00700")

    # 初始化cerebro回测系统设置
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MyStrategy)
    cerebro.adddata(data)

    # 设置初始资金和佣金
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.setcommission(commission=0.001)

    # 运行回测系统
    cerebro.run()

    # 得到回测结束后的总资金
    portvalue = cerebro.broker.getvalue()
    print(f"最终资金: {portvalue:.2f}元")
