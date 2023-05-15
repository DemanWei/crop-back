import os
import math
import datetime
import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.globals import ThemeType
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from src.utils.DataUtils import load_data_sql
from src.utils.DateUtils import get_date_range, datetime2str
from src.config.Config import *

import warnings

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
        data = load_data_sql(self.params, self.settings['miss_type'])
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
        # 日志
        info = "<activation function>:'relu'\n<epochs>:{}\n\n<loss>:{}".format(history.epoch, history.history['loss'])
        with open(CONSOLE_PRE_PATH + 'LSTM.txt', 'w', encoding='utf-8') as fp:
            fp.write(info)
        if self.settings['model_auto_save']:
            model.save(MODELS_PATH + 'LSTM.h5')
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
                .add_yaxis('Train Prediction', [round(x, 2) for x in train_predict.to_list()],
                           linestyle_opts=opts.LineStyleOpts(color='red'), is_symbol_show=False)
                .add_yaxis('Test Prediction', [round(x, 2) for x in test_predict.to_list()],
                           linestyle_opts=opts.LineStyleOpts(color='orange'), is_symbol_show=False)
                .add_yaxis('Future', [round(x, 2) for x in future_.to_list()],
                           linestyle_opts=opts.LineStyleOpts(color='green'), is_symbol_show=False)
                .set_global_opts(
                title_opts=opts.TitleOpts(
                    title='模型:LSTM  城市:{}  作物:{}  Train RMSE:{:.2f}  Test RMSE:{:.2f}'.format(self.params['city'],
                                                                                              self.params['crop'],
                                                                                              train_RMSE, test_RMSE),
                    subtitle="折线图"),
                tooltip_opts=opts.TooltipOpts(trigger='axis', axis_pointer_type='cross'),
                toolbox_opts=opts.ToolboxOpts(is_show=True, orient='vertical', pos_left='right'),
                legend_opts=opts.LegendOpts(pos_top=30)
            )
                .set_colors(['blue', 'red', 'orange', 'green'])
        )
        line.render(IMG_PRE_PATH + 'LSTM.html')


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
    datas = {'original': original, 'train_predict': train_predict, 'test_predict': test_predict, 'future': future}
    return datas, round(train_RMSE, 2), round(test_RMSE, 2), model
