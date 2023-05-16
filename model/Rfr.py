import datetime
import math
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from blueprint.data import get_price
from utils.DateUtils import get_date_range
from utils.echart_utils import render_predict

warnings.filterwarnings('ignore')


class Rfr:
    def __init__(self, params, conf, settings):
        self.params = params
        self.conf = conf
        self.settings = settings

    def load_data(self):
        """加载数据集,返回DataFrame格式"""
        # return load_data_sql(self.params, self.settings['miss_type'])
        # 加载city_crop的价格数据和图像
        data = get_price(self.params['city'], self.params['crop'])
        date_list = [x['date'] for x in data]
        price_list = [x['price'] for x in data]
        data = pd.DataFrame(price_list, index=date_list, columns=['price'])
        return data

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
            # todo save model
            joblib.dump(rf, './static/model/RFR.rf')
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
        render_predict('RFR', self.params, original, train_predict, test_predict, future, train_RMSE, test_RMSE,
                       save_path='./static/predict/RFR.html')


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
    datas = {'original': dict_to_str(data.to_dict()),
             'train_predict': dict_to_str(train_predict.to_dict()),
             'test_predict': dict_to_str(test_predict.to_dict()),
             'future': dict_to_str(future.to_dict())}
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