#  -*-coding:utf8 -*-
from selenium import webdriver
import time
import os
from selenium.webdriver.support.select import Select
browser = webdriver.Chrome()
browser.get("http://10.0.0.55/srun_portal_pc_yys.php?ac_id=8&")
try:
    elem_change = browser.find_element_by_name("portal-domain")
    Select(elem_change).select_by_value("@yidong")
except:
    pass
elem_user = browser.find_element_by_id("username")
elem_user.clear()
elem_user.send_keys("1120173330")
elem_pass = browser.find_element_by_id("password")
elem_pass.clear()
elem_pass.send_keys("wudawei120")
elem_login = browser.find_element_by_id("button")
elem_login.click()
time.sleep(1)
os.system("taskkill /f /t /im auto.exe")
