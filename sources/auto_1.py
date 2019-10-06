import win32api
from pykeyboard import *
k = PyKeyboard()
for i in range(50):
    win32api.ShellExecute(0, 'open', "Project2.exe", "", '', 1)
