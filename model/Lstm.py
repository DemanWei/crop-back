import datetime
import math
import os
import warnings

import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.models import Sequential, load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from blueprint.data import get_price
from utils.DateUtils import get_date_range
from utils.echart_utils import render_predict

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 防止显存不够(exit code -1073740791...)


class Lstm:
    def __init__(self, params, conf, settings):
        self.look_back = 1
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.params = params
        self.conf = conf
        self.settings = settings

    def load_data(self):
        # data = load_data_sql(self.params, self.settings['miss_type'])
        # 加载city_crop的价格数据和图像
        data = get_price(self.params['city'], self.params['crop'])
        date_list = [x['date'] for x in data]
        price_list = [x['price'] for x in data]
        data = pd.DataFrame(price_list, index=date_list, columns=['price'])
        return data

    def get_index(self, data):
        train_size = math.ceil(len(data) * self.settings['train_rate'])
        train = data[:train_size]
        test = data[train_size:]
        return train.index, test.index

    def normalize(self, data):
        """Min-Max归一化"""
        return self.scaler.fit_transform(data)

    def create_dataset(self, dataset):
        """将数组转化为矩阵"""
        # X: t时刻的乘客数, Y: t+1时刻的乘客数
        dataX, dataY = [], []
        for i in range(len(dataset) - self.look_back - 1):
            dataX.append(dataset[i:(i + self.look_back), 0])
            dataY.append(dataset[i + self.look_back, 0])
        return np.array(dataX), np.array(dataY)

    def split_data(self, data):
        """切分数据集"""
        train_size = math.ceil(len(data) * self.settings['train_rate'])
        # train = data[:train_size]
        # test = data[train_size:]
        train, test = data[0:train_size, :], data[train_size:, :]
        trainX, trainY = self.create_dataset(train)
        testX, testY = self.create_dataset(test)
        return trainX, trainY, testX, testY

    def reshape(self, trainX, testX):
        """reshape input to be [samples, time steps, features]"""
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        return trainX, testX

    def fit(self, trainX, trainY):
        '''训练模型'''
        # 创建 LSTM 网络
        model = Sequential()
        model.add(LSTM(self.conf['LSTM units'], input_shape=(1, self.look_back)))
        model.add(Dense(self.conf['Dense units']))
        model.compile(loss='mean_squared_error', optimizer='adam')
        history = model.fit(trainX, trainY, epochs=self.conf['epochs'], batch_size=self.conf['batch_size'],
                            verbose=self.conf['verbose'])
        # model = Sequential()
        # model.add(LSTM(20, input_shape=(1, self.look_back), dropout=0.0, kernel_regularizer=l1_l2(0.00, 0.00),bias_regularizer=l1_l2(0.00, 0.00)))
        # model.add(Dense(1))
        # model.compile(loss='mean_squared_error', optimizer='adam')
        # model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
        # TODO console 日志
        # info = "<activation function>:'relu'\n<epochs>:{}\n\n<loss>:{}".format(history.epoch, history.history['loss'])
        # with open(CONSOLE_PRE_PATH + 'LSTM.txt', 'w', encoding='utf-8') as fp:
        #     fp.write(info)
        if self.settings['model_auto_save']:
            # TODO save model
            model.save('./static/model/LSTM.h5')
        return model

    def predict(self, model, trainX, testX):
        """预测训练集和测试集"""
        return model.predict(trainX), model.predict(testX)

    def once_forecast(self, model, history, n_input):
        # flatten data
        data = np.array(history)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        # retrieve last observations for input data
        input_x = data[-n_input:, 0]
        # reshape into [1, n_input, 1]
        input_x = input_x.reshape((1, len(input_x), 1))
        # forecast the next week
        yhat = model.predict(input_x, verbose=0)
        # we only want the vector forecast
        yhat = yhat[0]
        return yhat

    def forecast(self, model, testX):
        """预测未来"""
        history = list(testX)
        future = []
        for i in range(self.params['limit']):
            yhat_sequence = self.once_forecast(model, history, 1)
            future.append(yhat_sequence)
            yhat_sequence = yhat_sequence.reshape(len(yhat_sequence), 1)
            history.append(yhat_sequence)
        return future

    def inverse(self, trainY, train_predict, testY, test_predict, future):
        """invert predictions"""
        train_predict = self.scaler.inverse_transform(train_predict)
        trainY = self.scaler.inverse_transform([trainY])
        test_predict = self.scaler.inverse_transform(test_predict)
        testY = self.scaler.inverse_transform([testY])
        future = self.scaler.inverse_transform(future)
        return trainY, train_predict, testY, test_predict, future

    def get_RMSE(self, trainY, train_predict, testY, test_predict):
        """评估"""
        train_RMSE = math.sqrt(mean_squared_error(trainY[0], train_predict[:, 0]))
        test_RMSE = math.sqrt(mean_squared_error(testY[0], test_predict[:, 0]))
        return train_RMSE, test_RMSE

    def do_plot(self, original, train_predict, test_predict, future, train_RMSE, test_RMSE):
        """绘图"""
        render_predict('LSTM', self.params, original, train_predict, test_predict, future, train_RMSE, test_RMSE,
                       save_path='./static/predict/LSTM.html')


def LSTM_main(params, conf, settings, model_path):
    lstm = Lstm(params, conf, settings)

    # 加载数据
    data_ = lstm.load_data()
    train_index, test_index = lstm.get_index(data_)
    np.random.seed(7)

    # 归一化
    data = lstm.normalize(data_)
    # 切分数据集
    trainX, trainY, testX, testY = lstm.split_data(data)  # 需要和get_index()的rate相同
    # reshape
    trainX, testX = lstm.reshape(trainX, testX)

    # 是否加载模型
    if model_path:
        model = load_model(model_path)
    else:
        model = lstm.fit(trainX, trainY)

    # 预测原数据
    train_predict, test_predict = lstm.predict(model, trainX, testX)
    # 预测未来
    future = lstm.forecast(model, testX)

    # inverse
    trainY, train_predict, testY, test_predict, future = lstm.inverse(trainY, train_predict, testY, test_predict,
                                                                      future)

    # 评估
    train_RMSE, test_RMSE = lstm.get_RMSE(trainY, train_predict, testY, test_predict)

    # 序列化为Series
    train_predict = pd.Series(train_predict.flatten(), index=train_index[:len(train_predict)])
    test_predict = pd.Series(test_predict.flatten(), index=test_index[:len(test_predict)])
    date_str = datetime.datetime.strftime(test_predict.index[-1], '%Y-%m-%d')
    future = pd.Series(future.flatten(), index=get_date_range(date_str, params['limit']))

    # 绘图
    lstm.do_plot(data_, train_predict, test_predict, future, train_RMSE, test_RMSE)

    # 返回
    original = pd.Series(data_['price'], index=data_.index)  # DataFrame -> Series
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