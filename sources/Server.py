#  -*-coding:utf8 -*-
import socket
def server():
    s = socket.socket()
    host = socket.gethostname()
    port = 12345
    s.bind((host, port))
    s.listen(5)

    while True:
        print("等待连接......")
        conn, addr = s.accept()
        print("连接地址:{}".format(addr))
        while True:
            try:
                recv_data = conn.recv(1024)
                print("服务器收到的信息:{}".format(recv_data.decode()))
            except:
                print("连接关闭")
                break

def main():
    server()

if __name__ == '__main__':
    main()