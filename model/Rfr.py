import math
import joblib
import datetime
import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.globals import ThemeType
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from src.utils.DataUtils import load_data_sql
from src.utils.DateUtils import get_date_range, datetime2str
from src.config.Config import *

import warnings

warnings.filterwarnings('ignore')


class Rfr:
    def __init__(self, params, conf, settings):
        self.params = params
        self.conf = conf
        self.settings = settings

    def load_data(self):
        """加载数据集,返回DataFrame格式"""
        return load_data_sql(self.params, self.settings['miss_type'])

    def split_data(self, data):
        train_size = math.ceil(len(data) * self.settings['train_rate'])
        X_train = data[:train_size]
        y_train = X_train['price']
        X_test = data[train_size:]
        return X_train, y_train, X_test

    def normalize(self, X_train, X_test):
        """归一化"""
        X_train_norm = (X_train - X_train.min()) / (X_train.max() - X_train.min())
        X_test_norm = (X_test - X_test.min()) / (X_test.max() - X_test.min())
        return X_train_norm, X_test_norm

    def fit(self, X_train_norm, y_train):
        """训练"""
        rf = RandomForestRegressor(max_features=self.conf['max_features'],
                                   min_samples_leaf=self.conf['min_samples_leaf'],
                                   n_estimators=self.conf['n_estimators'], random_state=self.conf['random_state'],
                                   oob_score=True)
        rf.fit(X_train_norm, y_train)
        if self.settings['model_auto_save']:
            joblib.dump(rf, MODELS_PATH + 'RFR.rf')
        return rf

    def predict(self, model, X_train_norm, X_test_norm):
        """预测训练集和测试集"""
        return model.predict(X_train_norm), model.predict(X_test_norm)

    def forecast(self, model, start):
        """预测未来"""
        new_index = get_date_range(start, self.params['limit'])
        meta = pd.Series(0, copy=True, index=new_index)
        indexs = meta.index
        from sklearn.preprocessing import StandardScaler
        meta = StandardScaler().fit_transform(meta.values.reshape(-1, 1))
        future = model.predict(meta)
        future = pd.Series(future, index=indexs)
        return future

    def get_RMSE(self, train_original, train_predict, test_original, test_predict):
        train_RMSE = math.sqrt(mean_squared_error(train_original, train_predict))
        test_RMSE = math.sqrt(mean_squared_error(test_original, test_predict))
        return train_RMSE, test_RMSE

    def do_plot(self, original, train_predict, test_predict, future, train_RMSE, test_RMSE):
        pre = pd.Series([np.nan for x in range(len(train_predict))])
        test_predict = pd.concat([pre, test_predict], axis=0)
        pre = pd.Series([np.nan for x in range(len(original))])
        future_ = pd.concat([pre, future], axis=0)
        new_index = pd.concat([original, future], axis=0)
        line = (
            Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width='792px', height='464px'))
                .add_xaxis([datetime2str(x) for x in new_index.index])
                .add_yaxis('Original', [round(x, 2) for x in list(original['price'])],
                           linestyle_opts=opts.LineStyleOpts(color='blue'), is_symbol_show=False)
                .add_yaxis('Train Prediction', [round(x, 2) for x in list(train_predict.values)],
                           linestyle_opts=opts.LineStyleOpts(color='red'), is_symbol_show=False)
                .add_yaxis('Test Prediction', [round(x, 2) for x in list(test_predict.values)],
                           linestyle_opts=opts.LineStyleOpts(color='orange'), is_symbol_show=False)
                .add_yaxis('Future', [round(x, 2) for x in list(future_.values)],
                           linestyle_opts=opts.LineStyleOpts(color='green'), is_symbol_show=False)
                .set_global_opts(
                title_opts=opts.TitleOpts(
                    title='模型:RFR  城市:{}  作物:{}  Train RMSE:{:.2f}  Test RMSE:{:.2f}'.format(self.params['city'],
                                                                                             self.params['crop'],
                                                                                             train_RMSE, test_RMSE),
                    subtitle="折线图"),
                tooltip_opts=opts.TooltipOpts(trigger='axis', axis_pointer_type='cross'),
                toolbox_opts=opts.ToolboxOpts(is_show=True, orient='vertical', pos_left='right'),
                legend_opts=opts.LegendOpts(pos_top=30)
            )
                .set_colors(['blue', 'red', 'orange', 'green'])
        )
        line.render(IMG_PRE_PATH + 'RFR.html')


def Rfr_main(params, conf, settings, model_path):
    rfr = Rfr(params, conf, settings)

    # 加载数据
    data = rfr.load_data()
    # 切分数据集
    X_train, y_train, X_test = rfr.split_data(data)
    # 归一化
    X_train_norm, X_test_norm = rfr.normalize(X_train, X_test)

    # 是否加载模型
    if model_path:
        model = joblib.load(model_path)
    else:
        model = rfr.fit(X_train_norm, y_train)
    # 预测原数据
    train_predict, test_predict = rfr.predict(model, X_train_norm, X_test_norm)
    # 预测未来
    date_str = datetime.datetime.strftime(X_test.index[-1] + datetime.timedelta(1), '%Y-%m-%d')
    future = rfr.forecast(model, date_str)

    # 评估
    train_RMSE, test_RMSE = rfr.get_RMSE(X_train['price'], train_predict, X_test['price'], test_predict)

    # 绘制
    train_predict = pd.Series(train_predict, index=X_train.index)
    test_predict = pd.Series(test_predict, index=X_test.index)
    rfr.do_plot(data, train_predict, test_predict, future, train_RMSE, test_RMSE)

    # 返回
    data = pd.Series(data['price'], index=data.index)  # DataFrame -> Series
    datas = {'original': data, 'train_predict': train_predict, 'test_predict': test_predict, 'future': future}
    return datas, round(train_RMSE, 2), round(test_RMSE, 2), model
