# 为了实现实时股票盯盘系统的数据存储模块，我们可以考虑以下几个函数：

# 1. connect_db(): 连接数据库，返回数据库连接对象。
# 2. create_table(conn): 创建数据表，接受一个数据库连接对象作为参数。
# 3. insert_data(conn, data): 将获取到的实时股票数据插入到数据表中，接受一个数据库连接对象和数据作为参数。

# 其中，connect_db() 函数连接数据库，可以使用 Python 内置的 sqlite3 模块实现。create_table(conn) 函数创建数据表，可以使用 SQL 语句实现。insert_data(conn, data) 函数将获取到的实时股票数据插入到数据表中，可以使用 SQL 语句和参数化查询实现。

# 具体实现可以参考以下代码：

# 其中，stock.db 是数据库文件名，可以根据实际情况修改。stock_data 是数据表名，包含股票代码、股票名称、股票价格和时间等字段。data 是一个元组，包含股票代码、股票名称、股票价格和时间等数据。

import sqlite3


def connect_db():
    conn = sqlite3.connect('stock.db')
    return conn

# todo 感觉数据库的列名不太对
def create_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code TEXT NOT NULL,
            stock_name TEXT NOT NULL,
            stock_price REAL NOT NULL,
            stock_time TEXT NOT NULL
        )
    ''')
    conn.commit()

def insert_data(conn, data):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO stock_data (stock_code, stock_name, stock_price, stock_time)
        VALUES (?, ?, ?, ?)
    ''', data)
    conn.commit()



if __name__ == '__main__':
    stock_code = "00700"

