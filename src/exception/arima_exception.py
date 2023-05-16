# @Author  : LZC
# @Time    : 2022/2/26 11:30
# @File    : arima_exception.py
# @Software : PyCharm
class ArimaDataTestFailedException(Exception):
    """ARIMA预测时,原数据检验不通过的异常"""

    def __init__(self, msg):
        """必须将msg传给super"""
        super().__init__(msg)  # 初始化父类
