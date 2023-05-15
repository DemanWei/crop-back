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
