import datetime

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from utils.hash_utils import encrypt

app = Flask(__name__, static_folder='../static')
ctx = app.app_context()
ctx.push()

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:105610@127.0.0.1:3306/crop'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_MAX_OVERFLOW'] = 5
app.config['SQLALCHEMY_POOL_SIZE'] = 20
app.config['SQLALCHEMY_POOL_TIMEOUT'] = 5
app.secret_key = 'Terraria'

db = SQLAlchemy(app)


class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    username = db.Column(db.String(256), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    create_time = db.Column(db.DateTime, nullable=False, default=datetime.datetime.now)
    status = db.Column(db.Integer, nullable=False, default=1)

    # 关系
    user_history = db.relationship('History', backref='user.id', cascade="all,delete")
    user_error = db.relationship('Error', backref='user.id', cascade="all,delete")

    def __repr__(self):
        desc = self.__class__.__name__ + '{'
        for field in self.keys():
            desc += '{}={},'.format(field, getattr(self, field))
        desc += '}'
        return desc

    @staticmethod
    def keys():
        # 屏蔽password字段
        return ['id', 'username', 'create_time', 'status']

    def __getitem__(self, item):
        return getattr(self, item)


class History(db.Model):
    __tablename__ = 'history'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    user_id = db.Column(db.BigInteger, db.ForeignKey('user.id', ondelete='CASCADE', onupdate='CASCADE'), nullable=False)
    city = db.Column(db.String(64))
    crop = db.Column(db.String(64))
    model = db.Column(db.String(32))
    freq = db.Column(db.String(8))
    consume_time = db.Column(db.Float)
    complex = db.Column(db.Integer)
    data_length = db.Column(db.Integer)
    train_rate = db.Column(db.Float)
    rmse = db.Column(db.String(32))
    create_time = db.Column(db.DateTime, nullable=False, default=datetime.datetime.now)
    status = db.Column(db.Integer, nullable=False, default=1)

    def __repr__(self):
        desc = self.__class__.__name__ + '{'
        for field in self.keys():
            desc += '{}={},'.format(field, getattr(self, field))
        desc += '}'
        return desc

    @staticmethod
    def keys():
        return ['id', 'user_id', 'city', 'crop', 'model', 'freq',
                'consume_time', 'complex', 'data_length',
                'train_rate', 'rmse', 'create_time', 'status']

    def __getitem__(self, item):
        return getattr(self, item)


class Error(db.Model):
    __tablename__ = 'error'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    user_id = db.Column(db.BigInteger, db.ForeignKey('user.id', ondelete='CASCADE', onupdate='CASCADE'), nullable=False)
    code = db.Column(db.String(64), unique=True, nullable=False)
    message = db.Column(db.String(512))
    create_time = db.Column(db.DateTime, nullable=False, default=datetime.datetime.now)
    status = db.Column(db.Integer, nullable=False, default=1)

    def __repr__(self):
        desc = self.__class__.__name__ + '{'
        for field in self.keys():
            desc += '{}={},'.format(field, getattr(self, field))
        desc += '}'
        return desc

    @staticmethod
    def keys():
        return ['id', 'user_id', 'code', 'message', 'create_time', 'status']

    def __getitem__(self, item):
        return getattr(self, item)


class Data(db.Model):
    __tablename__ = 'data'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    city = db.Column(db.String(64))
    crop = db.Column(db.String(64))
    date = db.Column(db.Date)
    price = db.Column(db.Float)

    def __repr__(self):
        desc = self.__class__.__name__ + '{'
        for field in self.keys():
            desc += '{}={},'.format(field, getattr(self, field))
        desc += '}'
        return desc

    @staticmethod
    def keys():
        return ['id', 'city', 'crop', 'date', 'price']

    def __getitem__(self, item):
        return getattr(self, item)

if __name__ == '__main__':
    if input('是否重置数据库?(y/n)') == 'y':
        # 重置数据库
        db.drop_all()
        # 新建
        db.create_all()
        # 生成数据
        user1 = User(username='90217', password=encrypt('123abc'), status=1)
        user2 = User(username='Azir', password=encrypt('123abc'), status=0)
        user3 = User(username='demanwei', password=encrypt('123abc'))
        user4 = User(username='90219', password=encrypt('123abc'))
        # 把数据提交给用户会话
        db.session.add_all([user1, user2, user3, user4])
        # 提交事务
        db.session.commit()
