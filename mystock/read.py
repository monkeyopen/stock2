import os
import csv
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
CONF_PATH = os.getenv('CONF_PATH')


class Stock:
    def __init__(self, name, price, volume, money):
        self.name = name
        self.price = float(price)
        self.volume = float(volume)
        self.money = float(money)

    def __str__(self):
        return f"{self.name},{self.price}, {self.volume}, {self.money}"


def read_us_data():
    label_file = CONF_PATH + "all_us_stock"
    num = 0
    allStock = []
    print(label_file)
    with open(label_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过第一行
        for arr in reader:
            print(arr)
            name = arr[4]
            price = arr[5]
            volume = arr[13]
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
    file_name = CONF_PATH + "us_stock"
    with open(file_name, 'w') as f:
        # 遍历筛选后的股票
        for stock in selected_stocks:
            # 将股票名称写入文件
            f.write(stock.name + '\n')


def read_hk_data():
    label_file = CONF_PATH + "all_hk_stock"
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
    label_file = CONF_PATH + "all_a_stock"
    num = 0
    allStock = []

    with open(label_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过第一行
        for arr in reader:
            # print(arr)
            name = arr[1]
            price = arr[3]
            volume = arr[12]
            money = arr[13]
            # money = float(price) * int(volume)
            # print(name, price, volume, money)
            stock = Stock(name, price, volume, money)
            allStock.append(stock)

    sorted_allStock = sorted(allStock, key=lambda stock: stock.money, reverse=True)
    selected_stocks = [stock for stock in sorted_allStock if stock.money > 5 * 1e8]
    selected_stocks = sorted(selected_stocks, key=lambda stock: stock.name)
    for stock in selected_stocks:
        print(stock.name)
    # 创建一个新的文件
    file_name = CONF_PATH + "a_stock"
    with open(file_name, 'w') as f:
        # 遍历筛选后的股票
        for stock in selected_stocks:
            # 将股票名称写入文件
            f.write(stock.name + '\n')


if __name__ == '__main__':
    # read_us_data()
    # read_hk_data()
    read_a_data()
