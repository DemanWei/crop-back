import datetime
import math
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.component.encoder import dict_to_str
from src.model.base_model import BaseModel
from src.utils.date_utils import get_date_range

warnings.filterwarnings('ignore')


class Rfr(BaseModel):
    def __init__(self, params, conf, settings):
        super().__init__('RFR', params, settings)
        self.conf = conf

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
        # console日志
        info = '''max_features: {}
max_depth: {}
max_leaf_nodes: {}
max_samples: {}
min_samples_leaf: {}
min_impurity_decrease: {}
min_samples_split: {}
min_weight_fraction_leaf: {}
n_estimators: {}
oob_score: {}'''.format(rf.max_features, rf.max_depth, rf.max_leaf_nodes, rf.max_samples,
                        rf.min_samples_leaf, rf.min_impurity_decrease, rf.min_samples_split,
                        rf.min_weight_fraction_leaf,
                        rf.n_estimators, rf.oob_score)
        self.console_log(info)

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


def Rfr_main(params, conf, settings, model_upload_name):
    rfr = Rfr(params, conf, settings)

    # 加载数据
    data = rfr.load_data()
    # 切分数据集
    X_train, y_train, X_test = rfr.split_data(data)
    # 归一化
    X_train_norm, X_test_norm = rfr.normalize(X_train, X_test)

    # 是否加载模型
    if model_upload_name is not None:
        model_upload_path = './static/model_upload/{}'.format(model_upload_name)
        model = joblib.load(model_upload_path)
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
    rfr.plot_res(data, train_predict, test_predict, future, train_RMSE, test_RMSE)

    datas = {
        'original': dict_to_str(data.to_dict()),
        'train_predict': dict_to_str(train_predict.to_dict()),
        'test_predict': dict_to_str(test_predict.to_dict()),
        'future': dict_to_str(future.to_dict())
    }
    return datas, round(train_RMSE, 2), round(test_RMSE, 2)
