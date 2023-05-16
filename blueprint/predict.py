import time

from flask import Blueprint, request, jsonify

bp_predict = Blueprint('predict', __name__, url_prefix='/predict')


@bp_predict.route('/predict', methods=['POST'])
def predict():
    model = request.json.get('model')
    params = request.json.get('params')
    conf = request.json.get('conf')
    settings = request.json.get('settings')
    model_upload_name = request.json.get('model_upload_name')

    if model is None or params is None:
        return jsonify(status=0, message='字段缺失', data=None)

    # 调用模型预测
    start_time = time.time()
    if model == 'ARIMA':
        from model.Arima import ARIMA_main
        datas, train_RMSE, test_RMSE = ARIMA_main(params, conf, settings, model_upload_name)
    elif model == 'LR':
        from model.Lr import LR_main
        datas, train_RMSE, test_RMSE = LR_main(params, conf, settings, model_upload_name)
    elif model == 'LSTM':
        from model.Lstm import LSTM_main
        datas, train_RMSE, test_RMSE = LSTM_main(params, conf, settings, model_upload_name)
    elif model == 'RFR':
        from model.Rfr import Rfr_main
        datas, train_RMSE, test_RMSE = Rfr_main(params, conf, settings, model_upload_name)
    else:
        return jsonify(status=0, message='参数值有误', data=None)
    # 训练预测总耗时
    all_time = time.time() - start_time
    summary = {'runtime': all_time, 'train_RMSE': train_RMSE, 'test_RMSE': test_RMSE}
    result = {
        'datas': datas,
        'summary': summary,
    }

    return jsonify(status=1, message='操作成功', data=result)
