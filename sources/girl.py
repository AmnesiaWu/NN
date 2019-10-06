import requests as rq
from bs4 import BeautifulSoup as bf
import os
if not os.path.exists("nidongde"):
    os.mkdir("nidongde")
counter = 0
for i in range(30):
    kv = {"user-agent" : "Mozilla/5.0"}
    r = rq.get("http://www.284yu.com/h0415/mp4list3/{}.htm".format(i), headers = kv)
    contents = bf(r.text, "html.parser")
    girls = contents('a')
    for girl in girls:
        try:
            link = girl.get("data-original")
            if link != None:
                temp = rq.get(link)
                with open("nidongde/{}.jpg".format(counter), 'wb') as f:
                    f.write(temp.content)
                    f.close()
                    print(counter)
                counter += 1
        except:
            pass
