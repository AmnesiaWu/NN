import xlrd
import xlwt
from xlutils.copy import copy
import pandas as pd
import numpy as np
import time
def whatever(label, img):
    t = time.time()
    data_frame = []
    for j, _ in enumerate(label):
        temp = label[j]
        temp_2 = img[j]
        temp.extend(temp_2)
        temp = tuple(temp)
        data_frame.append(temp)
    data = pd.DataFrame(data_frame, index=range(len(label)))
    data.to_csv(r'C:\Users\Administrator\Desktop\2.csv')
    print(time.time() - t)
    return data
label = np.random.randint(3, size=(10000, 1))
img = np.random.randint(3, size=(10000, 784))
label = label.tolist()
img = img.tolist()
whatever(label, img)