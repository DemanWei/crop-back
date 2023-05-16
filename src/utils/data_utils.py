# @Author  : LZC
# @Time    : 2021/8/11 10:30
# @File    : data_utils.py
# @Software : PyCharm
import pandas as pd
import pymysql

from src.config.db import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def get_duration_data(data, date_from, date_end):
    """获取每年指定时间段的数据"""
    new_data = data.head(0)
    years = data.index[-1].date().year - data.index[0].date().year + 1
    for i in range(years):
        year = data.index[0].date().year + i
        new_data = new_data.append(data[str(year) + '-' + date_from:str(year) + '-' + date_end])
    return new_data


def process_missing(data, type):
    """处理缺失值"""
    if type == 'drop':
        data.dropna(inplace=True)
    elif type == 'none':
        pass
    elif type == 'fill':
        data.fillna(axis=0, method='ffill', inplace=True)


"""==================================================================================="""

def load_data_sql(city, crop, freq='1D', miss_type='none', season_start=None, season_end=None):
    """从mysql加载数据"""
    sql = "select * from data where city='%s' and crop='%s'" % (city, crop)
    conn = pymysql.connect(host=IP, port=int(PORT), user=USERNAME, password=PASSWORD, database=DATABASE)
    data = pd.read_sql(sql, con=conn)
    # 取date和price两列,同时设置date为索引
    data.set_index('date', inplace=True)
    data = data[['price']]
    # 处理缺失数据
    try:
        data = data.resample(freq).mean()  # 重采样
    except Exception as _:
        pass
    process_missing(data, miss_type)
    # 季节轮换
    if season_start is not None and season_end is not None:
        data = get_duration_data(data, season_start, season_end)
    return data

if __name__ == '__main__':
    datas = load_data_sql('Lahore', 'Apple (Ammre)')
    print(datas)
