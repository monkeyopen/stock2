import os
import csv
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
CONF_PATH = os.getenv('CONF_PATH')


class Stock:
    def __init__(self, name, volume, money):
        self.name = name
        self.volume = float(volume)
        self.money = float(money)

    def __str__(self):
        return f"{self.name}, {self.volume:,.0f}, {self.money:,.0f}\n"


def read_us_data():
    list_file = CONF_PATH + "us_stock_20240215"
    num = 0
    allStock = []
    print(list_file)
    with open(list_file, 'r') as f:
        for line in f.readlines():
            stock = line.strip()
            if len(stock) < 1:
                continue
            print(stock)
            total_volume = 0
            total_money = 0
            stock_path = DATA_PATH + stock
            with open(stock_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # 跳过第一行

                for arr in reader:
                    # print(arr)
                    date = arr[1]
                    if date < "2024-02-01" or date > "2024-12-31":
                        continue
                    price = arr[5]
                    volume = arr[6]
                    money = float(price) * float(volume)
                    total_volume += float(volume)
                    total_money += money

            allStock.append(Stock(stock, total_volume, total_money))

    sorted_allStock = sorted(allStock, key=lambda stock: stock.money, reverse=True)
    selected_stocks = [stock for stock in sorted_allStock if stock.money > 1e8]

    # 创建一个新的文件
    file_name = CONF_PATH + "us_stock_20240215_selected_volume"
    with open(file_name, 'w') as f:
        # 遍历筛选后的股票
        for stock in selected_stocks[:1000]:
            # 将股票名称写入文件
            f.write(stock.__str__())

    selected_stocks = sorted(selected_stocks[:1000], key=lambda stock: stock.name)
    # 创建一个新的文件
    file_name = CONF_PATH + "us_stock_20240215_selected"
    with open(file_name, 'w') as f:
        # 遍历筛选后的股票
        for stock in selected_stocks:
            # 将股票名称写入文件
            f.write(stock.name + '\n')


def read_hk_data():
    label_file = CONF_PATH + "hk_stock_20240224"
    num = 0
    allStock = []

    with open(label_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过第一行
        for arr in reader:
            # print(arr)
            name = arr[1]
            price = arr[5]
            volume = arr[10]
            money = float(price) * int(volume)
            # print(name, price, volume, money)
            stock = Stock(name, price, volume, money)
            allStock.append(stock)

    sorted_allStock = sorted(allStock, key=lambda stock: stock.money, reverse=True)
    selected_stocks = [stock for stock in sorted_allStock if stock.money > 1e8]
    selected_stocks = sorted(selected_stocks, key=lambda stock: stock.name)
    for stock in selected_stocks:
        print(stock.name)
    # 创建一个新的文件
    file_name = CONF_PATH + "hk_stock"
    with open(file_name, 'w') as f:
        # 遍历筛选后的股票
        for stock in selected_stocks:
            # 将股票名称写入文件
            f.write(stock.name + '\n')


def read_a_data():
    list_file = CONF_PATH + "a_stock_20240219"
    num = 0
    allStock = []
    print(list_file)
    with open(list_file, 'r') as f:
        for line in f.readlines():
            stock = line.strip()
            if len(stock) < 1:
                continue
            print(stock)
            total_volume = 0
            total_money = 0
            stock_path = DATA_PATH + stock
            with open(stock_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # 跳过第一行

                for arr in reader:
                    # print(arr)
                    date = arr[1]
                    if date < "2024-02-01" or date > "2024-12-31":
                        continue
                    price = arr[5]
                    volume = arr[6]
                    money = float(price) * float(volume)
                    total_volume += float(volume)
                    total_money += money

            allStock.append(Stock(stock, total_volume, total_money))

    sorted_allStock = sorted(allStock, key=lambda stock: stock.money, reverse=True)
    selected_stocks = [stock for stock in sorted_allStock if stock.money > 1e8]

    # 创建一个新的文件
    file_name = CONF_PATH + "a_stock_20240219_selected_volume"
    with open(file_name, 'w') as f:
        # 遍历筛选后的股票
        for stock in selected_stocks[:]:
            # 将股票名称写入文件
            f.write(stock.__str__())

    selected_stocks = sorted(selected_stocks[:], key=lambda stock: stock.name)
    # 创建一个新的文件
    file_name = CONF_PATH + "a_stock_20240219_selected"
    with open(file_name, 'w') as f:
        # 遍历筛选后的股票
        for stock in selected_stocks:
            # 将股票名称写入文件
            f.write(stock.name + '\n')


if __name__ == '__main__':
    # read_us_data()
    read_hk_data()
    # read_a_data()

    print("Done!")
