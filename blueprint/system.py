import os
import uuid

from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename

from src.utils.file_utils import get_file_suffix

bp_system = Blueprint('system', __name__, url_prefix='/system')


@bp_system.route('/upload', methods=['POST'])
def upload():
    # 创建upload文件夹
    upload_dir = './static/model_upload/'
    if not os.path.exists(upload_dir):
        os.mkdir(upload_dir)
    # 获取请求参数
    user_id = request.form['user_id']
    file = request.files['upload_log_file']

    if user_id is None or file is None:
        return jsonify(status=0, message='字段缺失', data=None)

    # 生成uuid4
    global_uuid4 = uuid.uuid4()
    # 文件保存到服务器,这里要放前边
    file_path = '{}/{}_{}{}'.format(upload_dir, user_id, global_uuid4, get_file_suffix(secure_filename(file.filename)))
    file.save(file_path)
    # 响应
    data = {'model_upload_name': os.path.basename(file_path)}
    return jsonify(status=1, message='文件上传成功', data=data)


@bp_system.route('/download', methods=['POST'])
def download():
    file_path = request.json.get('file_path')
    # todo 限制只能访问static
    if file_path is None:
        return jsonify(status=0, message='字段缺失', data=None)

    # 虚拟目录 -> 相对文件目录
    file_path = '.' + file_path

    # 文件存在性和合法性校验
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return jsonify(status=0, message='未找到此文件', data=None)
    return send_file(os.path.abspath(file_path))
