# encoding:utf-8 #
from selenium import webdriver
import time
import winsound
import msvcrt
from selenium.webdriver.chrome.options import Options
from PIL import Image

class Course(object):
    def __init__(self, username, password,course_name, course_type):
        self.username = username
        self.password = password
        self.course_name = course_name
        self.type = course_type
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        self.browser = webdriver.Chrome(options=chrome_options)
        self.browser.get("https://login.bit.edu.cn/cas/login?service=http%3A%2F%2Fjwms.bit.edu.cn%2F")
    def Login(self):
        # 登录教务处选课系统
        elem_user = self.browser.find_element_by_id("username")
        elem_user.clear()
        elem_user.send_keys(self.username)
        elem_pass = self.browser.find_element_by_id("password")
        elem_pass.clear()
        elem_pass.send_keys(self.password)
        elem_login = self.browser.find_element_by_xpath('//input[@type="image"]')
        # self.browser.execute_script("$(arguments[0]).click()", elem_login)
        elem_login.click()
    def FindCourse(self):
        # 找到所选的课程
        elem_course_selection = self.browser.find_element_by_xpath('//div[@class="wap"]/a[1]')
        self.browser.execute_script("arguments[0].click()", elem_course_selection) # is not clickble at point solution
        elem_entry = self.browser.find_element_by_xpath('//a[text()="进入选课"]')
        elem_entry.click() # list
        elem_entry = self.browser.find_element_by_xpath('//a[text()="进入选课"]')
        elem_entry.click()  # list
        self.Switch_to_other_handle()
        if self.type == 2:
            elem_entry = self.browser.find_element_by_xpath('//a[text() = "公选课选课"]')
            url = elem_entry.get_attribute("href")
            js = 'window.open("{}");'.format(url)
            self.browser.execute_script(js)
            self.Switch_to_other_handle()
        else:
            elem_entry = self.browser.find_element_by_xpath('//a[text() = "本学期计划选课"]')
            url = elem_entry.get_attribute("href")
            js = 'window.open("{}");'.format(url)
            self.browser.execute_script(js)
            self.Switch_to_other_handle()
    def Select_course(self):
        # 判断是否可以选课
        if self.type == 2:# 公选课
            while True:
                points = '.'
                print("\r监视中{:3}".format(points), end='')
                time.sleep(0.5)
                elem_course_send = self.browser.find_element_by_xpath('//input[@id = "kcxx"]')
                elem_course_send.clear()
                elem_course_send.send_keys(self.course_name)
                elem_button = self.browser.find_element_by_xpath('//input[@type = "button"]')
                elem_button.click()
                points = '.' * 2
                print("\r监视中{:3}".format(points), end='')
                time.sleep(0.5)
                try:
                    elem_number_of_people = self.browser.find_element_by_xpath('//tr[@class="odd"]/td[11]')
                except:
                    continue
                points = '.' * 3
                print("\r监视中{:3}".format(points), end='')
                time.sleep(0.5)
                num = elem_number_of_people.text.split('/')
                current_num = int(num[0])
                if current_num > 0:
                    print('\n')
                    break
            winsound.Beep(1000, 3000)
            elem_select = self.browser.find_element_by_xpath('//tr[@class="odd"]/td[13]/a')
            elem_select.click()
            path = r'./1.png'
            self.browser.save_screenshot(path)
            img = Image.open(path)
            img.show()
            code = input("监视到课程可选，请输入选课验证码：")
            elem_send_code = self.browser.find_element_by_id('verifyCode')
            elem_send_code.send_keys(code)
            elem_submit = self.browser.find_element_by_id('changeVerifyCode')
            elem_submit.click()
            alert = self.browser.switch_to.alert
            print(alert.text)
            if alert.text != "选课成功！":
                alert.accept()
                self.Rerun()
                self.browser.close()
            else:
                self.browser.close()
        else:
            # 专业选修
            while True:
                points = '.'
                print("\r监视中{:3}".format(points), end='')
                time.sleep(0.5)
                elem_course_send = self.browser.find_element_by_xpath('//input[@id = "kcxx"]')
                elem_course_send.clear()
                elem_course_send.send_keys(self.course_name)
                elem_button = self.browser.find_element_by_xpath('//input[@value = "查询课程"]')
                elem_button.click()
                points = '.' * 2
                print("\r监视中{:3}".format(points), end='')
                time.sleep(0.5)
                # noinspection PyBroadException
                try:
                    self.browser.find_element_by_xpath('//tbody[@role="alert"]').click()
                except:
                    continue
                points = '.' * 3
                print("\r监视中{:3}".format(points), end='')
                time.sleep(1)
                try:
                    elem_number_of_people = self.browser.find_element_by_xpath('//tr[@class="odd"]/td[5]')
                except:
                    continue
                num = elem_number_of_people.text.split('/')
                current_num = int(num[0])
                total_num = int(num[1])
                if current_num < total_num:
                    print('\n')
                    break
            winsound.Beep(1000, 3000)
            elem_select = self.browser.find_element_by_xpath('//a[text()="选课"]')
            elem_select.click()
            path = r'./1.png'
            self.browser.save_screenshot(path)
            img = Image.open(path)
            img.show()
            code = input("监视到课程可选，请输入选课验证码：")
            img.close()
            elem_send_code = self.browser.find_element_by_id('verifyCode')
            elem_send_code.send_keys(code)
            elem_submit = self.browser.find_element_by_id('changeVerifyCode')
            elem_submit.click()
            alert = self.browser.switch_to.alert
            print(alert.text)
            if alert.text != "选课成功！":
                alert.accept()
                self.Rerun()
                self.browser.close()
            else:
                self.browser.close()
    def Switch_to_other_handle(self):
        # 切换窗口
        self.browser.close()
        handles = self.browser.window_handles
        self.browser.switch_to.window(handles[0])
    def Refresh(self):
        # 刷新
        self.browser.refresh()
    def Rerun(self):
        self.Refresh()
        self.Select_course()
    def Run(self):
        self.Login()
        self.FindCourse()
        self.Select_course()
def pwd_input():
    chars = []
    while True:
        # noinspection PyBroadException
        try:
            newChar = msvcrt.getch().decode(encoding="utf-8")
        except:
            return input("你很可能不是在cmd命令行下运行，密码输入将不能隐藏:")
        if newChar in '\r\n': # 如果是换行，则输入结束
             break
        elif newChar == '\b': # 如果是退格，则删除密码末尾一位并且删除一个星号
             if chars:
                 del chars[-1]
                 msvcrt.putch('\b'.encode(encoding='utf-8')) # 光标回退一格
                 msvcrt.putch( ' '.encode(encoding='utf-8')) # 输出一个空格覆盖原来的星号
                 msvcrt.putch('\b'.encode(encoding='utf-8')) # 光标回退一格准备接受新的输入
        else:
            chars.append(newChar)
            msvcrt.putch('*'.encode(encoding='utf-8')) # 显示为星号
    res = ''.join(chars)
    return res

if __name__ == '__main__':
    print("请输入学号：")
    username = input()
    print("请输入密码: ")
    password = pwd_input()
    print('\n')
    course_type = int(input("选择课程类型(1或2), 1:专业选修，2：公选课 ："))
    course_name = input("请输入需要选择的课程名：")
    # username = "1120173330"
    # password = "wudawei120"
    # course_name = "机器学习初步"
    # course_type = 1
    Crs = Course(username, password, course_name, course_type)
    Crs.Run()