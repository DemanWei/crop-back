from flask import Blueprint, request, jsonify

from src.domain.db_model import History, db
from src.utils.echart_utils import render_static

bp_history = Blueprint('history', __name__, url_prefix='/history')


@bp_history.route('/list', methods=['GET'])
def list():
    """ 列表 """
    user_id = request.args.get('user_id')
    # 所有
    if user_id is None:
        history_list = History.query.order_by(History.create_time.desc()).all()
        return jsonify(status=1, message='操作成功', data={
            'title': History.keys(),
            'history_list': history_list
        })
    # 某个user_id下的
    history_list = History.query.filter(History.user_id == user_id).order_by(History.create_time.desc()).all()
    return jsonify(status=1, message='操作成功', data={
        'title': History.keys(),
        'history_list': history_list
    })


@bp_history.route('/statistics', methods=['GET'])
def statistics():
    """统计"""
    user_id = request.args.get('user_id')
    if user_id is None:
        return jsonify(status=0, message='请求字段缺失', data=None)
    # sql = "select * from history where user_id='%d'" % (Global.user_id)
    history_list = History.query.filter(History.user_id == user_id).order_by(History.create_time.desc()).all()
    # 统计	{ARIMA,LR,RFR,LSTM}
    statistic = {'count': [0, 0, 0, 0], 'complex': [40, 20, 60, 80]}
    for item in history_list:
        model = item['model']
        if model == 'ARIMA':
            statistic['count'][0] += 1
        elif model == 'LR':
            statistic['count'][1] += 1
        elif model == 'RFR':
            statistic['count'][2] += 1
        elif model == 'LSTM':
            statistic['count'][3] += 1
    # 渲染柱状图
    render_static(statistic, save_path='./static/history/history.html')
    return jsonify(status=1, message='获取成功', data=statistic)


@bp_history.route('/delete', methods=['POST'])
def delete():
    """根据id删除history"""
    id = request.json.get('id')
    if id is None:
        return jsonify(status=0, message='请求字段缺失', data=None)

    history = History.query.get(id)
    db.session.delete(history)
    db.session.commit()
    return jsonify(status=1, message='操作成功', data=None)


@bp_history.route('/delete/all', methods=['POST'])
def deleteAll():
    """删除account的全部history"""
    user_id = request.json.get('user_id')
    if user_id is None:
        return jsonify(status=0, message='请求字段缺失', data=None)
    History.query.filter(History.user_id == user_id).delete(synchronize_session=False)
    db.session.commit()
    return jsonify(status=1, message='操作成功', data=None)


@bp_history.route('/add', methods=['POST'])
def add():
    """添加一条history"""
    user_id = request.json.get('user_id')
    city = request.json.get('city')
    crop = request.json.get('crop')
    model = request.json.get('model')
    freq = request.json.get('freq')
    consume_time = request.json.get('consume_time')
    complex = request.json.get('complex')
    data_length = request.json.get('data_length')
    train_rate = request.json.get('train_rate')
    rmse = request.json.get('rmse')
    if user_id is None:
        return jsonify(status=0, message='请求字段缺失', data=None)

    history = History(user_id=user_id, city=city, crop=crop, model=model, freq=freq,
                      consume_time=consume_time, complex=complex, data_length=data_length,
                      train_rate=train_rate, rmse=rmse)
    db.session.add(history)
    db.session.commit()
    return jsonify(status=1, message='操作成功', data=None)
    # sql = "insert into history(user_id,city,crop,model,freq,runtime,complex,data_length,train_rate,train_RMSE,test_RMSE,create_time)" \
    #       " values('%d','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')" % (
    #           user_id, record['params']['city'], record['params']['crop'],
    #           record['model'], record['params']['freq'], record['runtime'], record['complex'],
    #           record['data_length'], record['train_rate'],
    #           record['train_RMSE'], record['test_RMSE'], create_time
    #       )