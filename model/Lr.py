import datetime
import math
import warnings

import arrow
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

from blueprint.data import get_price
from utils.echart_utils import render_predict

warnings.filterwarnings('ignore')


class Lr:
    def __init__(self, params, settings):
        self.params = params
        self.settings = settings

    def load_data(self):
        # data = load_data_sql(self.params, self.settings['miss_type'])
        # 加载city_crop的价格数据和图像
        data = get_price(self.params['city'], self.params['crop'])
        date_list = [x['date'] for x in data]
        price_list = [x['price'] for x in data]
        data = pd.DataFrame(price_list, index=date_list, columns=['price'])
        return data

    def split_data(self, data, rate=0.9):
        """切分数据集"""
        train_size = math.ceil(len(data) * rate)
        train = data[:train_size]
        test = data[train_size:]
        return train, test

    def get_date_range(self, start, limit, level, format='YYYY-MM-DD'):
        """生成从start开始,间隔为level的日期序列"""
        start = arrow.get(start, format)
        res = (list(map(lambda dt: dt.format(format), arrow.Arrow.range(level, start, limit=limit))))
        date_parse2 = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
        return map(date_parse2, res)

    def fit(self, data, print_info=False):
        """训练"""
        print(data)
        print('=' * 30)
        print(type(data))
        slope, intercept, r, p, std_err = linregress(range(len(data)), data['price'])
        # TODO console日志
        # info = '回归公式: y = {}*X + {}\n'.format(slope, intercept)
        # with open(CONSOLE_PRE_PATH + 'LR.txt', 'w', encoding='utf-8') as fp:
        #     fp.write(info)
        # if print_info:
        #     print(info)
        # 保存模型
        if self.settings['model_auto_save']:
            # todo save model
            Lr.save_model('./static/model/LR.txt', slope, intercept)
        return slope, intercept

    def predict(self, train, test, slope, intercept):
        """预测训练集和测试集"""
        exp = range(len(train)) * slope + intercept
        train_predict = pd.Series(exp, train.index)
        test_predict_val = slope * range(len(train), len(train) + len(test)) + intercept
        test_predict = pd.Series(test_predict_val, test.index)
        return train_predict, test_predict

    def forest(self, data, slope, intercept):
        """预测未来"""
        start = str(data.index[-1].date() + datetime.timedelta(days=1))
        levels = {'1D': 'day', '7D': 'week', '1M': 'month', '1Y': 'year'}
        future_index = list(self.get_date_range(start, self.params['limit'], levels[self.params['freq']]))
        future_value = slope * range(len(data), len(data) + len(future_index)) + intercept
        future = pd.Series(future_value, future_index)
        return future

    def get_RMSE(self, train, train_predict, test, test_predict):
        """评估"""
        train_RMSE = math.sqrt(mean_squared_error(train, train_predict))
        test_RMSE = math.sqrt(mean_squared_error(test, test_predict))
        return train_RMSE, test_RMSE

    def plot_res(self, original, train_predict, test_predict, future, train_RMSE, test_RMSE):
        """绘图"""
        render_predict('LR', self.params, original, train_predict, test_predict, future, train_RMSE, test_RMSE,
                       save_path='./static/predict/LR.html')

    def load_model(self, path):
        """加载模型"""
        with open(path, 'r', encoding='utf-8') as fp:
            meta = fp.read().split('#')
            slope = np.array(float(meta[0]))
            intercept = np.array(float(meta[1]))
        return slope, intercept

    @staticmethod
    def save_model(model_path, slope, intercept):
        """保存训练模型"""
        with open(model_path, 'w', encoding='utf-8') as fp:
            fp.write('{}#{}'.format(slope, intercept))


def LR_main(params, conf, settings, model_path):
    """原数据"""
    lr = Lr(params, settings)
    # 加载数据
    data = lr.load_data()
    # 切分数据集
    train, test = lr.split_data(data, settings['train_rate'])

    if model_path:
        # 加载模型
        slope, intercept = lr.load_model(model_path)
    else:
        # 训练
        slope, intercept = lr.fit(train)
    # 预测原数据
    train_predict, test_predict = lr.predict(train, test, slope, intercept)
    # 预测未来
    future = lr.forest(data, slope, intercept)

    # 评估
    train_RMSE, test_RMSE = lr.get_RMSE(train, train_predict, test, test_predict)
    # 绘图
    lr.plot_res(data, train_predict, test_predict, future, train_RMSE, test_RMSE)

    # 返回
    datas = {'original': dict_to_str(data.to_dict()),
             'train_predict': dict_to_str(train_predict.to_dict()),
             'test_predict': dict_to_str(test_predict.to_dict()),
             'future': dict_to_str(future.to_dict())}
    fit = {'slope': slope, 'intercept': intercept}
    return datas, round(train_RMSE, 2), round(test_RMSE, 2)


def dict_to_str(data_dict):
    ret = {}
    for k, v in data_dict.items():
        if isinstance(v, dict):
            v = dict_to_str(v)
        elif isinstance(v, pd.DataFrame):
            v = v.to_dict()
        elif isinstance(v, pd.Timestamp):
            v = v.strftime('%Y-%m-%d')
        ret[str(k)] = v
    return ret