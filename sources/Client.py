# -*-coding:utf8 -*-
import socket

def client():
    s = socket.socket()
    host = socket.gethostname()
    port = 12345
    s.connect((host, port))

    while True:
        send_data = input()
        s.send(send_data.encode('utf-8'))

def main():
    client()

if __name__ == '__main__':
    main()