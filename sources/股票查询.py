import requests as rq
from bs4 import BeautifulSoup as bs
import re

def get_post(url):
    post_res = rq.get(url)
    post = []
    soup = bs(post_res.text, 'html.parser')
    a = soup.find_all('a')
    for att in a:
        link = att.get('href')
        try:
            temp_post = re.findall(r's[hz]\d{6}', link)
            if temp_post:
                post.append(temp_post)
        except:
            continue
    return post


def get_stock_info(stock_url, stock_refer_url):
    post = get_post(stock_refer_url)
    for i in range(len(post)):
        dic = {}
        temp = stock_url
        try:
            temp += post[i][0] + '.html'
            res = rq.get(temp)
            res.encoding = res.apparent_encoding
            soup = bs(res.text, 'html.parser')
            name = soup.find_all(attrs={'class': 'bets-name'})[0]
            dic.update({'股票名称': name.text.split()[0]})
            all_keys = soup.find_all('dt')
            all_values = soup.find_all('dd')
            for temp in range(len(all_keys)):
                dic[all_keys[temp].text] = all_values[temp].text
            with open('D:/stock.txt', 'a', encoding='utf-8') as f:
                f.write(str(dic) + '\n')
            print("\r{:.3f}%".format(((i + 1) / len(post)) * 100), end='')
        except:
            pass
def main():
    stock_url = 'https://gupiao.baidu.com/stock/'
    stock_refer_url = 'http://quote.eastmoney.com/stock_list.html'
    get_stock_info(stock_url, stock_refer_url)


if __name__ == '__main__':
    main()