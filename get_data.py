# 在get_data.py文件中实现get_stock_data函数
# 使用pandas-datareader库获取指定股票代码的实时股票数据，并将其存储在一个Pandas DataFrame中
# 如果指定了update_interval参数，则使用Python中的定时器技术定时获取最新的股票数据，并将其存储在DataFrame中
# 如果指定了file_path参数，则将获取到的实时股票数据存储到指定的文件中
# 使用异步编程或多线程技术来提高数据获取的效率

# 以下是可以用于获取美股、港股和A股交易数据的Python库：
# 1. pandas-datareader：这个库提供了一个简单的方式来获取Yahoo Finance、Google Finance等数据源的数据。可以使用pandas-datareader获取美股和港股的数据。
# 2. tushare：这个库提供了一个简单的方式来获取A股的数据。它可以获取股票、指数、基金、期货等数据。
# 3. yfinance：这个库提供了一个简单的方式来获取Yahoo Finance的数据。可以使用yfinance获取美股的数据。
# 4. akshare：这个库提供了一个简单的方式来获取A股、港股和美股的数据。它可以获取股票、指数、基金、期货等数据。
# 这些库都是开源的，可以在GitHub上找到它们的源代码。如果你想使用这些库，你需要先安装它们。你可以使用pip来安装它们，例如：
# pip install pandas-datareader
# pip install tushare
# pip install yfinance
# pip install akshare
# 安装完成后，你就可以在你的Python代码中使用它们来获取交易数据了。


# 最开始的时候可能是按股票代码拉取全部历史数据。后面每天更新的时候，只要拉取当天的全部数据，然后写入数据库就可以了。


import akshare as ak
import pandas as pd
import threading
import time


def get_stock_data(stock_code: str, update_interval: int = None, file_path: str = None) -> pd.DataFrame:
    """
    获取指定股票代码的历史股票数据，并将其存储在一个Pandas DataFrame中。
    如果指定了update_interval参数，则使用Python中的定时器技术定时获取最新的股票数据，并将其存储在DataFrame中。
    如果指定了file_path参数，则将获取到的实时股票数据存储到指定的文件中。
    """
    # def update_data():
    #     nonlocal df
    #     while True:
    #         new_data = ak.stock_hk_hist(symbol=stock_code, adjust="qfq")
    #         df = pd.concat([df, new_data])
    #         time.sleep(update_interval)
    df = ak.stock_hk_daily(symbol=stock_code, adjust="qfq")
    print(df)
    # if update_interval:
    #     t = threading.Thread(target=update_data)
    #     t.start()
    if file_path:
        df.to_csv(file_path)
    return df


def read_stock_data(file_path: str):
    """
    将CSV格式的记录读取为Pandas DataFrame
    """
    df = pd.read_csv(file_path)
    return df


if __name__ == '__main__':
    stock_code = "09888"
    df = get_stock_data(stock_code, file_path=f"data/{stock_code}")
    print(df)