# encoding:utf-8 #
from copy import deepcopy

def rain(li):
    newli = deepcopy(li)
    size = len(newli)
    # 从左向右填充雨水
    for i in list(range(size - 1)):
        for j in list(range(i + 1, size)):
            if newli[j] >= newli[i]:
                newli[i + 1:j] = [newli[i]] * (j - i - 1)
                break
    # 从右向左填充雨水
    for i in list(range(size - 1, 0, -1)):
        for j in list(range(i - 1, -1, -1)):
            if newli[j] >= newli[i]:
                newli[j + 1:i] = [newli[i]] * (i - j - 1)
                break
    return newli


if __name__ == '__main__':
    li = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    result = rain(li)
    print(result)
    print((sum(result) - sum(li)))