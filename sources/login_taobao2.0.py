# -*- coding : utf-8 -*-
import requests as rq
import re
import os
from win32crypt import CryptUnprotectData
import sqlite3


def main():
    counter = 1
    rq1 = rq.session()
    url = "https://s.taobao.com/search?q="
    words = input("输入需要搜索的物品:")
    url = url + words
    cookies = login_taobao_get_cookie()
    kv = {'cookie':cookies, 'user-agent':'Mozilla/5.0'}
    t = rq1.get(url, headers=kv)
    t.raise_for_status()
    t.encoding = t.apparent_encoding
    plt = re.findall(r'"view_price":"[\d.]*"', t.text)
    tlt = re.findall(r'"raw_title":".*?"', t.text)
    record = []
    for k in range(len(plt)):
        price = eval(plt[k].split(':')[-1])
        title = tlt[k].split(':')[-1]
        record.append([price, title])
    print("{:^4}\t{:^8}\t{:^18}".format("序号", "价格", '名称'))
    for j in record:
        print("{:^4}\t{:^8}\t{:^10}".format(counter, j[0], j[1]))
        counter += 1


def login_taobao_get_cookie():
    host = '.taobao.com'
    cookies_str = ''
    cookiepath = os.environ['LOCALAPPDATA'] + r"\Google\Chrome\User Data\Default\Cookies"
    sql = "select host_key,name,encrypted_value from cookies where host_key='%s'" % host
    with sqlite3.connect(cookiepath) as conn:
        cu = conn.cursor()
        cookies = {name: CryptUnprotectData(encrypted_value)[1].decode() for host_key, name, encrypted_value in
                   cu.execute(sql).fetchall()}
        for key, values in cookies.items():
            cookies_str = cookies_str + str(key) + "=" + str(values) + ';'
        return cookies_str

if __name__ == '__main__':
    main()