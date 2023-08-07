import tushare as ts


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = ts.get_hist_data('600519')
    print(df)
    df = ts.get_hist_data('00700')
    print(df)
    df = ts.get_hist_data('TQQQ')
    print(df)
