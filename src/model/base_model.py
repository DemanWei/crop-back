import math

from sklearn.metrics import mean_squared_error

from src.utils.data_utils import load_data_sql
from src.utils.echart_utils import render_predict


class BaseModel():
    def __init__(self, model, params, settings):
        self.params = params
        self.settings = settings
        self.model = model

    def load_data(self):
        """加载数据集,返回DataFrame格式"""
        return load_data_sql(self.params['city'], self.params['crop'], self.params['freq'],
                             self.settings['miss_type'])

    def split_data(self, data, rate=0.9):
        """切分数据集"""
        pass

    def fit(self, trainX, trainY):
        '''训练模型'''
        pass

    def predict(self, model, trainX, testX):
        """预测训练集和测试集"""
        pass

    def forecast(self, data, slope, intercept):
        """预测未来"""
        pass

    def get_RMSE(self, train, train_predict, test, test_predict):
        """评估"""
        train_RMSE = math.sqrt(mean_squared_error(train, train_predict))
        test_RMSE = math.sqrt(mean_squared_error(test, test_predict))
        return train_RMSE, test_RMSE

    def plot_res(self, original, train_predict, test_predict, future, train_RMSE, test_RMSE):
        """绘图"""
        render_predict(self.model, self.params, original, train_predict, test_predict, future, train_RMSE, test_RMSE,
                       save_path='./static/predict/ARIMA.html')

    def console_log(self, info):
        log_path = './static/log/{}.log'.format(self.model)
        with open(log_path, 'w', encoding='utf-8') as fp:
            fp.write(info)