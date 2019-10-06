import pytesseract
from PIL import Image as ig
import tensorflow as tf
import requests as rq
kv = {"user-agent": "Mozilla/5.0"}
temp = rq.get("http://img.ivsky.com/img/tupian/t/201901/11/bandiangou-006.jpg", headers=kv)
print(temp.content)
