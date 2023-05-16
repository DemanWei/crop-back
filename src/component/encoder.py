import datetime
import decimal
import uuid
import pandas as pd

from flask.json import JSONEncoder as _JSONEncoder


class JSONEncoder(_JSONEncoder):
    def default(self, o):
        if hasattr(o, 'keys') and hasattr(o, '__getitem__'):
            return dict(o)
        elif isinstance(o, datetime.datetime):
            return o.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(o, datetime.date):
            return o.strftime('%Y-%m-%d')
        elif isinstance(o, pd.Timestamp):
            return o.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(o, decimal.Decimal):
            return str(o)
        elif isinstance(o, uuid.UUID):
            return str(o)
        elif isinstance(o, bytes):
            return o.decode('utf-8')
        else:
            _JSONEncoder.default(self, o)


def dict_to_str(data_dict):
    """将pandas.TimeStamp的key转化为string"""
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