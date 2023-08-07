import pandas as pd

if __name__ == '__main__':
    # 读取股票数据
    df = pd.read_csv('data/00700')

    # 计算收盘价的9日移动平均线
    df['ma9'] = df['close'].rolling(window=9).mean()

    # 判断是否出现九转上涨结构
    for i in range(8, len(df)):
        if all(df.loc[i-8:i, 'close'] > df.loc[i-8:i, 'ma9']):
            print('九转上涨结构出现在', df.loc[i, 'date'])

    # 判断是否出现九转下跌结构
    for i in range(8, len(df)):
        if all(df.loc[i-8:i, 'close'] < df.loc[i-8:i, 'ma9']):
            print('九转下跌结构出现在', df.loc[i, 'date'])



