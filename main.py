from linear import linear_SVM

from Guass import guass_SVM
from linear import linear_SVM1

from Guass import guass_SVM1

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    linear_SVM("Iris/iris.data")
    print("----------------------分界线-----------------------")
    guass_SVM("Iris/iris.data")
    print("----------------------分界线-----------------------")
    linear_SVM1("wine/wine.data")
    print("----------------------分界线-----------------------")
    guass_SVM1("wine/wine.data")
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
