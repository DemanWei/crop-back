import math
import warnings
import datetime
import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.globals import ThemeType
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMAResults, ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.http.QThreadHttp import QThreadHttpPost
from src.utils.ExceptionUtils import ArimaDataTestFailedException
from src.utils.DataUtils import load_data_sql
from src.utils.DateUtils import get_date_range, datetime2str
from src.config.Config import *
from src.utils.decorator import auto_handle_error

warnings.filterwarnings('ignore')


class Arima():
    def __init__(self, params, settings):
        self.params = params
        self.settings = settings
        self.log = ''

    def load_data(self):
        # 加载city_crop的价格数据和图像
        return load_data_sql(self.params, self.settings['miss_type'])

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
        self.console_log(CONSOLE_PRE_PATH + 'ARIMA.txt', self.log)
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
        if self.settings['model_auto_save']:
            fit.save(MODELS_PATH + 'ARIMA.pkl')
        return fit

    def ARIMA_Predict(self, fit, data_test):
        """预测训练集和测试集"""
        train_predict = fit.predict()
        predict_ts = fit.forecast(len(data_test))
        test_predict = pd.Series(predict_ts.values, copy=True, index=data_test.index)
        return train_predict, test_predict

    def ARIMA_Forecast(self, fit, start):
        """预测未来"""
        predict_ts = fit.forecast(self.params['limit'])
        new_index = get_date_range(start, self.params['limit'])
        future = pd.Series(predict_ts.values, copy=True, index=new_index)  # 生成未来时间序列
        return future

    def get_RMSE(self, train, train_predict, test, test_predict):
        train_RMSE = math.sqrt(mean_squared_error(train, train_predict))
        test_RMSE = math.sqrt(mean_squared_error(test, test_predict))
        return train_RMSE, test_RMSE

    # return 40.2, 200.1

    def plot_res(self, original, train_predict, test_predict, future, train_RMSE, test_RMSE):
        """数据可视化"""
        pre = pd.Series([np.nan for x in range(len(train_predict))])
        test_predict = pd.concat([pre, test_predict], axis=0)
        pre = pd.Series([np.nan for x in range(len(original))])
        future_ = pd.concat([pre, future], axis=0)

        new_index = pd.concat([original, future], axis=0)
        line = (
            Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width='792px', height='464px'))
                .add_xaxis([datetime2str(x) for x in new_index.index])
                .add_yaxis('Original', [round(x, 2) for x in original.values.flatten()],
                           linestyle_opts=opts.LineStyleOpts(color='blue'), is_symbol_show=False)
                .add_yaxis('Train Prediction', [round(x, 2) for x in train_predict.to_list()],
                           linestyle_opts=opts.LineStyleOpts(color='red'), is_symbol_show=False)
                .add_yaxis('Test Prediction', [round(x, 2) for x in test_predict.tolist()],
                           linestyle_opts=opts.LineStyleOpts(color='orange'), is_symbol_show=False)
                .add_yaxis('Future', [round(x, 2) for x in future_.to_list()],
                           linestyle_opts=opts.LineStyleOpts(color='green'), is_symbol_show=False)
                .set_global_opts(
                title_opts=opts.TitleOpts(
                    title='模型:ARIMA  城市:{}  作物:{}  Train RMSE:{:.2f}  Test RMSE:{:.2f}'.format(self.params['city'],
                                                                                               self.params['crop'],
                                                                                               train_RMSE, test_RMSE),
                    subtitle="折线图"),
                tooltip_opts=opts.TooltipOpts(trigger='axis', axis_pointer_type='cross'),
                toolbox_opts=opts.ToolboxOpts(is_show=True, orient='vertical', pos_left='right'),
                legend_opts=opts.LegendOpts(pos_top=30)
            )
                .set_colors(['blue', 'red', 'orange', 'green'])
        )
        line.render(IMG_PRE_PATH + 'ARIMA.html')

    def console_log(self, console_path, info):
        with open(console_path, 'w', encoding='utf-8') as fp:
            fp.write(info)


def ARIMA_main(params, conf, settings, model_path):
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
        if model_path:
            fit = ARIMAResults.load(model_path)
        else:
            fit = arima.fit(train, (p, d, q))
        # 保存fit
        if settings.get('model_auto_save'):
            fit.save(MODELS_PATH + 'ARIMA.pkl')
        # 原数据预测
        train_predict, test_predict = arima.ARIMA_Predict(fit, test)
        # 预测未来
        date_str = datetime.datetime.strftime(test.index[-1] + datetime.timedelta(1), '%Y-%m-%d')
        future = arima.ARIMA_Forecast(fit, date_str)
        # 获取误差
        train_RMSE, test_RMSE = arima.get_RMSE(train, train_predict, test, test_predict)
        # 绘制
        arima.plot_res(day_data, train_predict, test_predict, future, train_RMSE, test_RMSE)

        # 返回
        datas = {'original': day_data, 'train_predict': train_predict, 'test_predict': test_predict, 'future': future}
        return datas, round(train_RMSE, 2), round(test_RMSE, 2), fit
    else:
        raise ArimaDataTestFailedException(
            'The stationarity or white noise test failed, please differentiate again or change the sampling frequency!'
        )

# # save model
# model_fit.save('model.pkl')
# # load model
# loaded = ARIMAResults.load('model.pkl')
