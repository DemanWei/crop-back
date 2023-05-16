import os

# 修改工作路径,覆盖pycharm默认的
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from flask import jsonify

from src.component.encoder import JSONEncoder
from src.domain.db_model import app

from blueprint.user import bp_user
from blueprint.history import bp_history
from blueprint.error import bp_error
from blueprint.system import bp_system
from blueprint.data import bp_data
from blueprint.predict import bp_predict

# json序列化model
app.json_encoder = JSONEncoder

# 模块化路由
app.register_blueprint(bp_user)
app.register_blueprint(bp_system)
app.register_blueprint(bp_history)
app.register_blueprint(bp_error)
app.register_blueprint(bp_data)
app.register_blueprint(bp_predict)


@app.route('/')
def hello_world():
    return jsonify(status=1, message='Hello World!', data=None)


if __name__ == '__main__':
    app.run(debug=False, port=5000)
