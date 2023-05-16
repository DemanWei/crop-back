import datetime
import math
import warnings

import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMAResults, ARIMA
from statsmodels.tsa.stattools import adfuller as ADF

from src.component.encoder import dict_to_str
from src.exception.arima_exception import ArimaDataTestFailedException
from src.model.base_model import BaseModel
from src.utils.date_utils import get_date_range

warnings.filterwarnings('ignore')


class Arima(BaseModel):
    def __init__(self, params, settings):
        super().__init__('ARIMA', params, settings)
        self.log = ''

    def split_data(self, data):
        train_size = math.ceil(len(data) * self.settings['train_rate'])
        train = data[:train_size]
        test = data[train_size:]
        return train, test

    def check_stationarity(self, data, print_info=False):
        """平稳性和白噪声检验,检验通过返回True"""
        # 平稳性检验——单位根检验: pvalue<0.05为平稳序列
        pvalue = ADF(data)[1]
        test_pass = pvalue < 0.05
        # console记录
        l0 = "-" * 18 + '平稳性检验' + '-' * 19 + '\n'
        l1 = 'pvalue={}\t为平稳序列:{}\n'.format(pvalue, test_pass)
        if test_pass:
            l2 = '检验[通过]!\n' + '-' * 45 + '\n'
        else:
            l2 = '检验[未通过]!!!\n' + '-' * 45 + '\n'
        info = l0 + l1 + l2
        self.log += info
        # 控制台打印
        if print_info:
            print(info)
        return test_pass

    def check_whiteNoise(self, data, print_info=False):
        """白噪声检验"""
        # 白噪声序列检验: accor<0.05为非白噪声序列
        accor = acorr_ljungbox(data, lags=1, return_df=False)[1][0]
        test_pass = accor < 0.05

        # console记录
        l0 = "-" * 18 + '白噪声检验' + '-' * 19 + '\n'
        l1 = 'accor={}\t非白噪声序列:{}\n'.format(accor, test_pass)
        if test_pass:
            l2 = '检验[通过]!\n' + '-' * 45 + '\n'
        else:
            l2 = '检验[未通过]!!!\n' + '-' * 45 + '\n'
        info = l0 + l1 + l2
        self.log += info
        # console log
        self.console_log(self.log)
        # 控制台打印
        if print_info:
            print(info)
        return test_pass

    def diff_data(self, data, d=1, plot=True):
        """绘制数据data的d阶差分图,并返回差分后的数据"""
        data_diff = data.diff(d)
        data_diff = data_diff.dropna()
        return data_diff

    def plot_ACF_PACF(self, data, lags=20):
        """绘制ACF与PACF图"""
        acf = fig.add_subplot(211)
        fig = plot_acf(data, lags=lags, ax=acf)
        acf.xaxis.set_ticks_position('bottom')
        acf.set_title('ACF')
        fig.tight_layout()
        pacf = fig.add_subplot(212)
        fig = plot_pacf(data, lags=lags, ax=pacf)
        pacf.xaxis.set_ticks_position('bottom')
        pacf.set_title('PACF')
        fig.tight_layout()

    def fit(self, train, order):
        """训练"""
        fit = ARIMA(train, order=order).fit()
        if self.settings.get('model_auto_save'):
            # TODO model save
            fit.save('./static/model/ARIMA.pkl')
        return fit

    def predict(self, fit, data_test):
        """预测训练集和测试集"""
        train_predict = fit.predict()
        predict_ts = fit.forecast(len(data_test))
        test_predict = pd.Series(predict_ts.values, copy=True, index=data_test.index)
        return train_predict, test_predict

    def forecast(self, fit, start):
        """预测未来"""
        predict_ts = fit.forecast(self.params['limit'])
        new_index = get_date_range(start, self.params['limit'])
        future = pd.Series(predict_ts.values, copy=True, index=new_index)  # 生成未来时间序列
        return future




def ARIMA_main(params, conf, settings, model_upload_name):
    """入口函数"""
    arima = Arima(params, settings)
    day_data = arima.load_data()
    # 1阶差分(d=1)
    data_diff = arima.diff_data(day_data, d=1, plot=False)

    # 平稳性检验通过,才可使用ARIMA
    if arima.check_stationarity(day_data) and arima.check_whiteNoise(day_data):
        # ACF和PACF确定q和p
        # arima.plot_ACF_PACF(day_data)
        # 调用ARIMA模型
        p = conf['p']
        d = conf['d']
        q = conf['q']
        # split data
        train, test = arima.split_data(day_data)

        # 是否加载模型
        if model_upload_name is not None:
            model_upload_path = './static/model_upload/{}'.format(model_upload_name)
            fit = ARIMAResults.load(model_upload_path)
        else:
            fit = arima.fit(train, (p, d, q))

        # 原数据预测
        train_predict, test_predict = arima.predict(fit, test)
        # 预测未来
        date_str = datetime.datetime.strftime(test.index[-1] + datetime.timedelta(1), '%Y-%m-%d')
        future = arima.forecast(fit, date_str)
        # 获取误差
        train_RMSE, test_RMSE = arima.get_RMSE(train, train_predict, test, test_predict)
        # 绘制
        arima.plot_res(day_data, train_predict, test_predict, future, train_RMSE, test_RMSE)

        # 返回
        datas = {
            'original': dict_to_str(day_data.to_dict()),
            'train_predict': dict_to_str(train_predict.to_dict()),
            'test_predict': dict_to_str(test_predict.to_dict()),
            'future': dict_to_str(future.to_dict())
        }
        return datas, round(train_RMSE, 2), round(test_RMSE, 2)
    else:
        raise ArimaDataTestFailedException(
            'The stationarity or white noise test failed, please differentiate again or change the sampling frequency!'
        )

# # save model
# model_fit.save('model.pkl')
# # load model
# loaded = ARIMAResults.load('model.pkl')
