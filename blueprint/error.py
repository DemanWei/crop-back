from flask import Blueprint, request, jsonify

from src.domain.db_model import Error, db

bp_error = Blueprint('error', __name__, url_prefix='/error')


@bp_error.route('/submit', methods=['POST'])
def submit_error():
    """错误信息和时间记录到数据库"""
    user_id = request.json.get('user_id')
    code = request.json.get('code')
    message = request.json.get('message')

    if user_id is None or code is None or message is None:
        return jsonify(status=0, message='请求字段缺失', data=None)

    error = Error(user_id=user_id, code=code, message=message)
    db.session.add(error)
    db.session.commit()
    return jsonify(status=1, message='操作成功', data=None)
    # sql = "insert into error(user_id,code,message,create_time) values('%d','%s','%s','%s')" % (
    #     Global.user_id, code, escape_string(message), create_time
    # )
