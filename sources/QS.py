from datetime import datetime
import numpy as np
def partition(arr, low, high):
    i = (low - 1)  # 最小元素索引
    pivot = arr[high]

    for j in range(low, high):
        # 当前元素小于或等于 pivot
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quickSort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)

arr = np.random.randint(1, 100000, 10000)
n = len(arr)
start = datetime.now()
quickSort(arr, 0, n - 1)
end = datetime.now()
res = (end - start).microseconds/1000
print('{}ms'.format(res))