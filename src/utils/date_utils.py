# @Author  : LZC
# @Time    : 2021/10/18 14:04
# @File    : date_utils.py
# @Software : PyCharm
import datetime

import arrow
import pandas as pd


def get_date_range(start, limit, level='day', format='YYYY-MM-DD'):
    """生成从start开始,间隔为day的日期序列"""
    start = arrow.get(start, format)
    res = (list(map(lambda dt: dt.format(format), arrow.Arrow.range(level, start, limit=limit))))
    date_parse2 = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    return map(date_parse2, res)


def datetime2str(time, format='%Y-%m-%d'):
    return datetime.datetime.strftime(time, format)
