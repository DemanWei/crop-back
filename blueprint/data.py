from flask import Blueprint, request, jsonify

from src.component.encoder import dict_to_str
from src.utils.data_utils import load_data_sql
from src.domain.db_model import Data
from src.utils.echart_utils import render_echarts

bp_data = Blueprint('data', __name__, url_prefix='/data')


@bp_data.route('/table', methods=['POST'])
def get_price():
    """获取某city某crop价格数据"""
    city = request.json.get('city')
    crop = request.json.get('crop')
    # 参数合法性校验
    if city is None or crop is None:
        return jsonify(status=0, message='请求字段缺失', data=None)
    # 查询数据库
    price_list = Data.query.filter(Data.city == city, Data.crop == crop).all()
    data = {'price_list': price_list, 'title': Data.keys()}

    # 返回数据
    return jsonify(status=1, message='获取成功', data=data)

@bp_data.route('/price', methods=['POST'])
def price():
    """获取某city某crop价格数据"""
    city = request.json.get('city')
    crop = request.json.get('crop')
    freq = request.json.get('freq')
    miss_type = request.json.get('miss_type')
    season_start = request.json.get('season_start')
    season_end = request.json.get('season_end')
    # 参数合法性校验
    if city is None or crop is None:
        return jsonify(status=0, message='请求字段缺失', data=None)

    # 查询数据库
    # data = Data.query.filter(Data.city == city, Data.crop == crop).all()
    # 使用pandas查询直接得到DataFrame
    data = load_data_sql(city, crop, freq, miss_type, season_start, season_end)
    # 渲染图像
    with_echarts = freq is not None
    if with_echarts:
        render_echarts(data, request.json, save_path='./static/echarts/original_data.html')

    # 返回数据
    data = None if with_echarts else dict_to_str(data.to_dict())
    return jsonify(status=1, message='获取成功', data=data)





@bp_data.route('/cityWithCrop', methods=['POST'])
def city_with_crop():
    """根据city获取所有crop"""
    city = request.json.get('city')
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

