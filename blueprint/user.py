import re

from flask import Blueprint, request, jsonify

from utils.db_model import User, db
from utils.hash_utils import encrypt

bp_user = Blueprint('user', __name__, url_prefix='/user')


@bp_user.route('/get', methods=['GET'])
def user():
    """根据username查询"""
    username = request.args.get('username')
    # 字段合法性校验
    if username is None:
        return jsonify(status=0, message='请求字段缺失', data=None)
    user = User.query.filter(User.username == username).first()
    return jsonify(status=1, message='获取成功', data=user)


@bp_user.route('/login', methods=['POST'])
def login():
    """ 登录 """
    username = request.json['username']
    password = request.json['password']
    # 请求字段缺失
    if username is None or password is None:
        return jsonify(status=0, message='请求字段缺失', data=None)
    target_user = User.query.filter(User.username == username, User.password == encrypt(password)).first()
    # 没查到用户
    if target_user is None:
        return jsonify(status=-1, message='用户名或密码错误', data=None)
    # 用户被禁用
    if target_user.status == 0:
        return jsonify(status=-1, message='该账户被冻结,请联系管理员', data=None)
    # 登录成功
    return jsonify(status=1, message='登陆成功', data=target_user)


@bp_user.route('/register', methods=['POST'])
def register():
    """ 注册 """
    # 获取请求参数
    username = request.json['username']
    password = request.json['password']
    password2 = request.json['password2']
    # 字段完整性校验
    if username is None or password is None or password is None:
        return jsonify(status=0, message='字段缺失', data=None)
    # 字段合法性校验
    if len(username) == 0 or len(password) == 0 or len(password2) == 0:
        return jsonify(status=0, message='字段有误', data=None)
    # 两次密码不一致
    if password != password2:
        return jsonify(status=-1, message='两次输入的密码不一致', data=None)
    # 用户已被注册
    if User.query.filter(User.username == username).first() != None:
        return jsonify(status=-1, message='该用户已注册', data=None)
    # 密码格式不符合要求
    if not re.findall('^(?![0-9]+$)(?![a-zA-Z]+$)[0-9A-Za-z]{6,18}$', password):
        return jsonify(status=-1, message='密码格式不合要求(6-18位字母和数字的组合)', data=None)
    # 注册成功
    db.session.add(User(
        username=username,
        password=encrypt(password)
    ))
    db.session.commit()
    return jsonify(status=1, message='注册成功', data=None)


@bp_user.route('/list', methods=['GET'])
def list_user():
    """ 列表/分页查询/模糊查询 """
    current = request.args.get('current', type=int)
    size = request.args.get('size', type=int)
    username = request.args.get('username')

    if username and len(username):
        if current and size:
            page_obj = User.query.filter(User.username.like('%{}%'.format(username))) \
                .order_by(User.create_time.desc()) \
                .paginate(page=current, per_page=size, error_out=False)

            return jsonify(status=1, message='获取成功', data={
                'title': User.keys(),
                'user_list': page_obj.items,  # page_obj.items: 返回当前页的内容列表
                'current': current,
                'size': size,
                'pages': page_obj.pages,
                'has_next': page_obj.has_next,
                'has_prev': page_obj.has_prev
            })
        else:
            user_list = User.query.filter(User.username.like('%{}%'.format(username))).all()
            return jsonify(status=1, message='获取成功', data={
                'title': User.keys(),
                'user_list': user_list
            })
    else:
        if current and size:
            page_obj = User.query.order_by(User.create_time.desc()) \
                .paginate(page=current, per_page=size, error_out=False)
            return jsonify(status=1, message='获取成功', data={
                'title': User.keys(),
                'user_list': page_obj.items,  # page_obj.items: 返回当前页的内容列表
                'current': current,
                'size': size,
                'pages': page_obj.pages,
                'has_next': page_obj.has_next,
                'has_prev': page_obj.has_prev
            })
        else:
            user_list = User.query.all()
            return jsonify(status=1, message='获取成功', data={
                'title': User.keys(),
                'user_list': user_list
            })


@bp_user.route('/modifyPassword', methods=['POST'])
def modify_password():
    """ 修改密码 """
    user_id = request.json.get('id')
    password_original = request.json.get('password_original')
    password_new = request.json.get('password_new')
    password_new2 = request.json.get('password_new2')
    if user_id is None or password_original is None or password_new is None or password_new2 is None:
        return jsonify(status=0, message='字段缺失', data=None)
    target_user = User.query.get(user_id)
    if target_user is None:
        return jsonify(status=-1, message='未找到该用户', data=None)
    if target_user.password != encrypt(password_original):
        return jsonify(status=-1, message='原密码错误', data=None)
    if password_new != password_new2:
        return jsonify(status=-1, message='两次输入的密码不一致', data=None)
    if not re.findall('^(?![0-9]+$)(?![a-zA-Z]+$)[0-9A-Za-z]{6,18}$', password_new):
        return jsonify(status=-1, message='密码格式不合要求(6-18位字母和数字的组合)', data=None)
    target_user.password = encrypt(password_new)
    db.session.commit()
    return jsonify(status=1, message='修改成功', data=None)


@bp_user.route('/delete', methods=['POST'])
def delete():
    user_id = request.json.get('id')
    if user_id is None:
        return jsonify(status=0, message='字段缺失', data=None)
    user = User.query.get(user_id)
    db.session.delete(user)
    db.session.commit()
    return jsonify(status=1, message='操作成功', data=None)


@bp_user.route('/modify', methods=['POST'])
def modify():
    user_id = request.json.pop('id')
    if user_id is None:
        return jsonify(status=0, message='字段缺失', data=None)
    # 传来什么字段就修改哪个
    user = User.query.get(user_id)
    if user is None:
        return jsonify(status=-1, message='未找到该用户', data=None)
    for field, value in request.json.items():
        setattr(user, field, value)
    db.session.commit()
    return jsonify(status=1, message='操作成功', data=None)
