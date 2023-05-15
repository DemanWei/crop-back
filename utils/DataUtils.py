# @Author  : LZC
# @Time    : 2021/8/11 10:30
# @File    : DataUtils.py
# @Software : PyCharm
import pandas as pd

from src.utils import DBUtils

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
# def load_data_sql(params, miss_type):
#     """从mysql加载数据"""
#     sql = "select * from data where city='%s' and crop='%s'" % (params['city'], params['crop'])
#     data = DBUtils.read_by_pd(sql)
#     # 取date和price两列,同时设置date为索引
#     data.set_index('date', inplace=True)
#     data = data[['price']]
#     # 处理缺失数据
#     try:
#         data = data.resample(params['freq']).mean()  # 重采样
#     except Exception as e:
#         pass
#     process_missing(data, miss_type)
#     # 季节轮换
#     if params.get('season_start') and params.get('season_end'):
#         data = get_duration_data(data, params['season_start'], params['season_end'])
#     return data

