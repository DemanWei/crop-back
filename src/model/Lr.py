import datetime
import math
import warnings

import arrow
import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.component.encoder import dict_to_str
from src.model.base_model import BaseModel

warnings.filterwarnings('ignore')


class Lr(BaseModel):
    def __init__(self, params, settings):
        super().__init__('LR', params, settings)

    def split_data(self, data):
        """切分数据集"""
        train_size = math.ceil(len(data) * self.settings['train_rate'])
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
        slope, intercept, r, p, std_err = linregress(range(len(data)), data['price'])
        # console日志
        info = '回归公式: y = {}*X + {}\n'.format(slope, intercept)
        self.console_log(info)
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

    @staticmethod
    def load_model(path):
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


def LR_main(params, conf, settings, model_upload_name):
    """原数据"""
    lr = Lr(params, settings)
    # 加载数据
    data = lr.load_data()
    # 切分数据集
    train, test = lr.split_data(data)

    if model_upload_name is not None:
        # 加载模型
        model_upload_path = './static/model_upload/{}'.format(model_upload_name)
        slope, intercept = Lr.load_model(model_upload_path)
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
    datas = {
        'original': dict_to_str(data.to_dict()),
        'train_predict': dict_to_str(train_predict.to_dict()),
        'test_predict': dict_to_str(test_predict.to_dict()),
        'future': dict_to_str(future.to_dict())
    }
    return datas, round(train_RMSE, 2), round(test_RMSE, 2)
