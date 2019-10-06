# -*- coding : utf-8 -*-
import requests as rq
import re


def main():
    counter = 1
    rq1 = rq.session()
    url = "https://s.taobao.com/search?q="
    words = input("输入需要搜索的物品:")
    url = url + words
    login_taobao('18212320927', 'wudawei120', rq1)
    t = rq1.get(url)
    t.raise_for_status()
    t.encoding = t.apparent_encoding
    plt = re.findall(r'"view_price":"[\d.]*"', t.text)
    tlt = re.findall(r'"raw_title":".*?"', t.text)
    record = []
    for k in range(len(plt)):
        price = eval(plt[k].split(':')[-1])
        title = tlt[k].split(':')[-1]
        record.append([price, title])
    print("{:6}\t{:8}\t{:10}".format("序号", "名称", '价格'))
    for j in record:
        print("{:6}\t{:8}\t{:10}".format(counter, j[1], j[0]))
        counter += 1


def login_taobao(username, password, rq1):
    temp = 'https://login.taobao.com/member/login.jhtml?spm=a21bo.2017.754894437.1.5af911d93Nyuhv&f=top&redirectURL=https%3A%2F%2Fwww.taobao.com%2F'
    headers = {
        'origin': 'http://login.taobao.com',
        'referer':temp,
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36',
    }
    postdata = {
        'TPL_username':username,
        'TPL_password' : password
    }
    url = 'https://login.taobao.com/member/login.jhtml?spm=a21bo.2017.754894437.1.5af911d93Nyuhv&f=top&redirectURL=https%3A%2F%2Fwww.taobao.com%2F'
    print(rq1.post(url, data= postdata, headers = headers).text)

if __name__ == '__main__':
    main()
