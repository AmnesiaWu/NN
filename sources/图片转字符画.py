#-*- coding:utf-8 -*-
import PIL.Image as ig

ascii_char = list(r"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")


def get_char(r, g, b, alpha = 255):
    if alpha == 0:
        return ' '
    length = len(ascii_char)
    gray = int(r * 0.2126 + g * 0.7152 + b * 0.0722)#三色转灰度
    unit = (256.0 + 1) / length#归一化
    return ascii_char[int(gray / unit)]


def main():
    file_name = input("请输出图片路径:")
    width = eval(input("请输入图片的最终宽度:"))
    height = eval(input("请输入图片的最终高度:"))
    im = ig.open(file_name)
    im = im.resize((width, height), ig.NEAREST)
    txt = ''
    for i in range(height):
        for j in range(width):
            txt += get_char(*im.getpixel((j, i)))
        txt += '\n'
    print(txt)


if __name__ == "__main__":
    main()