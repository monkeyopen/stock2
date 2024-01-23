import os
import csv
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')


class Stock:
    def __init__(self, name, price, volume, money):
        self.name = name
        self.price = float(price)
        self.volume = float(volume)
        self.money = float(money)

    def __str__(self):
        return f"{self.name},{self.price}, {self.volume}, {self.money}"


def read_us_data():
    label_file = DATA_PATH + "all_us_stock"
    num = 0
    allStock = []

    with open(label_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过第一行
        for arr in reader:
            # print(arr)
            name = arr[4]
            price = arr[5]
            volume = arr[13]
            money = float(price) * int(volume)
            # print(name, price, volume, money)
            stock = Stock(name, price, volume, money)
            allStock.append(stock)

    sorted_allStock = sorted(allStock, key=lambda stock: stock.money, reverse=True)
    for stock in sorted_allStock[:200]:
        print(stock)


def read_hk_data():
    label_file = DATA_PATH + "all_hk_stock"
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
    for stock in sorted_allStock[:100]:
        print(stock)


def read_a_data():
    label_file = DATA_PATH + "all_a_stock"
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
    for stock in sorted_allStock[:100]:
        print(stock)


if __name__ == '__main__':
    # read_us_data()
    # read_hk_data()
    read_a_data()
