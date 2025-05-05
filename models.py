from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(100), unique=True)
    role = db.Column(db.String(20), default='user', nullable=False)  # 'user'或'admin'
    created_at = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp())
    updated_at = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())
    
    @property
    def is_admin(self):
        return self.role == 'admin'

class Image(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.String(36), primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    url = db.Column(db.String(255), nullable=False)
    original_url = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp())
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'))
    embedding = db.Column(db.LargeBinary)
    
    captions = db.relationship('Caption', backref='image', lazy=True, cascade='all, delete-orphan')
    tags = db.relationship('Tag', secondary='image_tags', backref=db.backref('images', lazy=True))

class Tag(db.Model):
    __tablename__ = 'tags'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

class ImageTag(db.Model):
    __tablename__ = 'image_tags'
    image_id = db.Column(db.String(36), db.ForeignKey('images.id', ondelete='CASCADE'), primary_key=True)
    tag_id = db.Column(db.Integer, db.ForeignKey('tags.id', ondelete='CASCADE'), primary_key=True)

class Caption(db.Model):
    __tablename__ = 'captions'
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.String(36), db.ForeignKey('images.id', ondelete='CASCADE'))
    caption = db.Column(db.Text, nullable=False)

class Favorite(db.Model):
    __tablename__ = 'favorites'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'))
    image_id = db.Column(db.String(36), db.ForeignKey('images.id', ondelete='CASCADE'))
    added_at = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp())
    
    __table_args__ = (db.UniqueConstraint('user_id', 'image_id'),)

class Download(db.Model):
    __tablename__ = 'downloads'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'))
    image_id = db.Column(db.String(36), db.ForeignKey('images.id', ondelete='CASCADE'))
    downloaded_at = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp())

# 用户检索记录模型
class SearchHistory(db.Model):
    __tablename__ = 'search_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    search_content = db.Column(db.String(255), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.now)
    
    # 关联用户
    user = db.relationship('User', backref=db.backref('search_history', lazy=True))
    
    def __repr__(self):
        return f"<SearchHistory {self.id}: {self.search_content}>"

# 用户浏览记录模型
class ViewHistory(db.Model):
    __tablename__ = 'view_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_id = db.Column(db.String(36), db.ForeignKey('images.id', ondelete='CASCADE'), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.now)
    
    # 关联用户和图片
    user = db.relationship('User', backref=db.backref('view_history', lazy=True))
    image = db.relationship('Image', backref=db.backref('views', lazy='dynamic'))
    
    def __repr__(self):
        return f"<ViewHistory {self.id}: User {self.user_id} - Image {self.image_id}>"

# 新闻数据模型
class News(db.Model):
    __tablename__ = 'news'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=True)  # 标题可为空
    content = db.Column(db.Text, nullable=False)      # 内容不可为空
    topic = db.Column(db.String(20), nullable=False)  # 主题类别
    sentiment = db.Column(db.String(10), nullable=False)  # 情感分类：正面/中性/负面
    upload_time = db.Column(db.DateTime, default=datetime.now)  # 上传时间
    embedding = db.Column(db.LargeBinary, nullable=True)  # 嵌入向量，使用二进制存储
    
    def __repr__(self):
        return f"<News {self.id}: {self.title or '无标题'}>"