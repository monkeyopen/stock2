# 为了实现实时股票盯盘系统的账户管理模块，我们可以考虑以下几个函数：
#
# 1. registeraccount(): 注册账户，接受用户名和密码作为参数。
# 2. 为了实现实时股票盯盘系统的账户管理模块，我们可以考虑以下几个函数：
#
# 1. registeraccount(): 注册账户，接受用户名和密码作为参数。
# 2. loginaccount(): 登录账户，接受用户名和密码作为参数。
# 3. logoutaccount(): 注销账户，接受用户名作为参数。
#
# 其中，registeraccount() 函数用于注册账户，可以将用户名和密码存储在数据库中。loginaccount() 函数用于登录账户，可以检查用户名和密码是否匹配，并返回登录状态。logoutaccount() 函数用于注销账户，可以将用户的登录状态设置为未登录。
#
# 具体实现可以参考以下代码：(): 登录账户，接受用户名和密码作为参数。
# 3. logoutaccount(): 注销账户，接受用户名作为参数。
#
# 其中，registeraccount() 函数用于注册账户，可以将用户名和密码存储在数据库中。loginaccount() 函数用于登录账户，可以检查用户名和密码是否匹配，并返回登录状态。logoutaccount() 函数用于注销账户，可以将用户的登录状态设置为未登录。
#
# 具体实现可以参考以下代码：
#
# 其中，user.db 是数据库文件名，可以根据实际情况修改。user 是数据表名，包含用户名、密码和登录状态等字段。registeraccount() 函数将用户名和密码插入到数据表中。loginaccount() 函数检查用户名和密码是否匹配，并将登录状态设置为已登录。logoutaccount() 函数将登录状态设置为未登录。




import sqlite3

def connect_db():
    conn = sqlite3.connect('user.db')
    return conn

def create_table(conn):
    sql = '''
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            is_login INTEGER NOT NULL DEFAULT 0
        );
    '''
    conn.execute(sql)
    conn.commit()

def register_account(conn, username, password):
    sql = '''
        INSERT INTO user (username, password) VALUES (?, ?);
    '''
    conn.execute(sql, (username, password))
    conn.commit()

def login_account(conn, username, password):
    sql = '''
        SELECT * FROM user WHERE username = ? AND password = ?;
    '''
    cursor = conn.execute(sql, (username, password))
    row = cursor.fetchone()
    if row:
        sql = '''
            UPDATE user SET is_login = 1 WHERE id = ?;
        '''
        conn.execute(sql, (row[0],))
        conn.commit()
        return True
    else:
        return False

def logout_account(conn, username):
    sql = '''
        UPDATE user SET is_login = 0 WHERE username = ?;
    '''
    conn.execute(sql, (username,))
    conn.commit()



