from flask import Blueprint, request, jsonify

from utils.db_model import Data

bp_data = Blueprint('data', __name__, url_prefix='/data')


@bp_data.route('/price')
def price():
    """获取某city某crop价格数据"""
    city = request.args.get('city')
    crop = request.args.get('crop')
    # 参数合法性校验
    if city is None or crop is None:
        return jsonify(status=0, message='请求字段缺失', data=None)
    # 返回数据
    price_list = Data.query.filter(Data.city == city, Data.crop == crop).all()
    return jsonify(status=1, message='获取成功', data=price_list)


@bp_data.route('/cityWithCrop')
def city_with_crop():
    """根据city获取所有crop"""
    city = request.args.get('city')
    # 参数合法性校验
    if city is None:
        return jsonify(status=0, message='请求字段缺失', data=None)
    # 返回数据
    city_list = Data.query.with_entities(Data.city.distinct()).all()
    distinct_citys = [row[0] for row in city_list]
    crop_list = Data.query.with_entities(Data.crop.distinct()).filter(Data.city == city).all()
    distinct_crops = [row[0] for row in crop_list]
    data = {'city_list': distinct_citys, 'crop_list': distinct_crops}
    return jsonify(status=1, message='获取成功', data=data)
