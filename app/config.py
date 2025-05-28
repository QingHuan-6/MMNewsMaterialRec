# backend/config.py
import os
from datetime import timedelta
import logging  # 导入 logging 模块

# 定义 'backend' 文件夹的基准目录
# 假设 config.py 直接位于 'backend' 文件夹内
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

class Config:
    """基础配置类，包含所有环境共有的配置。"""
    # Flask 核心配置
    # 重要: SECRET_KEY 必须是一个复杂、随机且保密的字符串！
    # 对于生产环境，强烈建议从 instance/config.py 或环境变量加载。
    SECRET_KEY = os.environ.get('SECRET_KEY') or '这是一个用于开发的默认密钥-请务必在生产中更改'
    DEBUG = False
    TESTING = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB 上传文件大小限制

    # 服务器配置
    SERVER_HOST = os.environ.get('SERVER_HOST') or 'localhost'
    SERVER_PORT = int(os.environ.get('SERVER_PORT') or 5000)
    SERVER_IP = f"{SERVER_HOST}:{SERVER_PORT}"  # 用于URL生成（替代原app.py中的Server_IP）

    # JWT Extended 配置
    # 重要: JWT_SECRET_KEY 同样需要保密！
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or '这是另一个JWT开发密钥-生产中务必更改'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=30)  # Token 过期时间

    # SQLAlchemy 数据库配置
    # 生产环境的数据库连接字符串应通过 instance/config.py 或环境变量提供
    DB_USER = os.environ.get('DB_USER', 'root')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', '123456')  # 开发默认密码
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_NAME = os.environ.get('DB_NAME', 'pixtock')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
                              f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}?charset=utf8mb4&init_command=SET%20time_zone%3D%27%2B08%3A00%27"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 文件/目录路径名称 (这些将在 __init__.py 中基于 app.root_path 或 BASE_DIR 构建)
    CNA_FOLDER_NAME = 'CNA_images'
    CACHE_DIR_NAME = 'CNA_cache'
    THUMBNAIL_FOLDER_NAME = 'thumbnails'
    UPLOAD_FOLDER_NAME = 'uploads'
    FLICKR_FOLDER_NAME = 'Flickr8K\\Images'
    CAPTIONS_FILE_NAME = 'Flickr8K\\captions.txt'

    # CN-CLIP 模型配置
    CN_CLIP_MODEL_NAME = "ViT-B-16"  # 例如: "ViT-B-16" 或 "RN50"

    # 缩略图配置
    THUMBNAIL_SIZE = (200, 200)

    # 日志配置 (可以在 __init__.py 中进一步配置)
    LOG_LEVEL = logging.ERROR  # 应用的默认日志级别

    @staticmethod
    def init_app(app):
        """允许在应用创建后执行额外的配置初始化。"""
        # 根据上面定义的名称构建实际路径并存入 app.config
        # 使用 app.instance_path 可以定位到 instance 文件夹同级的目录 (通常是 backend/)
        # 或者使用 BASE_DIR (config.py 所在的目录)
        # 为确保路径的正确性，通常建议在 app 创建后，基于 app.root_path (app包的路径) 或 app.instance_path (instance文件夹的父目录) 来构建

        # 更稳妥的做法是在 app/__init__.py 中，当 app 对象可用时，使用 app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, '..', app.config['UPLOAD_FOLDER_NAME'])
        # 或者直接使用 BASE_DIR (即 backend/ 目录)
        app.config['CNA_FOLDER'] = os.path.join(BACKEND_DIR, app.config['CNA_FOLDER_NAME'])
        app.config['CACHE_DIR'] = os.path.join(BACKEND_DIR, app.config['CACHE_DIR_NAME'])
        app.config['IMAGE_EMBED_CACHE'] = os.path.join(app.config['CACHE_DIR'], "CNA_image_embeddings.npz")
        app.config['THUMBNAIL_FOLDER'] = os.path.join(BACKEND_DIR, app.config['THUMBNAIL_FOLDER_NAME'])
        app.config['UPLOAD_FOLDER'] = os.path.join(BACKEND_DIR, app.config['UPLOAD_FOLDER_NAME'])
        app.config['FLICKR_FOLDER'] = os.path.join(BACKEND_DIR, app.config['FLICKR_FOLDER_NAME'])
        app.config['CAPTIONS_FILE'] = os.path.join(BACKEND_DIR, app.config['CAPTIONS_FILE_NAME'])

        # 确保目录存在
        os.makedirs(app.config['CNA_FOLDER'], exist_ok=True)
        os.makedirs(app.config['CACHE_DIR'], exist_ok=True)
        os.makedirs(app.config['THUMBNAIL_FOLDER'], exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        if not os.path.exists(os.path.dirname(app.config['FLICKR_FOLDER'])):
            os.makedirs(os.path.dirname(app.config['FLICKR_FOLDER']), exist_ok=True)
            
        app.logger.info(f"上传文件夹配置于: {app.config['UPLOAD_FOLDER']}")


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    LOG_LEVEL = logging.DEBUG
    # SQLALCHEMY_ECHO = True # 对于调试SQL查询很有用
    # 开发时可以使用一个简单、固定的 SECRET_KEY，但切勿用于生产
    SECRET_KEY = 'dev-secret-key-for-flask'
    JWT_SECRET_KEY = 'dev-jwt-secret-key'


class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    # 测试时也可能需要Debug模式
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL') or 'sqlite:///:memory:'  # 测试时使用内存数据库
    WTF_CSRF_ENABLED = False  # 测试表单时通常禁用CSRF保护
    # 为测试设置固定的密钥
    SECRET_KEY = 'test-secret-key'
    JWT_SECRET_KEY = 'test-jwt-secret'


class ProductionConfig(Config):
    """生产环境配置"""
    # 生产环境中，SECRET_KEY 和 DATABASE_URL 必须通过环境变量或 instance/config.py 设置
    # SECRET_KEY = os.environ.get('SECRET_KEY') # 从系统环境变量获取
    # JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') # 从系统环境变量获取
    # SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') # 从系统环境变量获取

    # 注意: 如果不使用系统环境变量，这些值必须在 instance/config.py 中定义
    # Flask 会自动从 instance/config.py 加载并覆盖这里的设置（如果 instance/config.py 存在且包含这些键）

    # 示例检查 (如果依赖环境变量且未设置则抛出错误)
    # if not Config.SECRET_KEY and not os.environ.get('SECRET_KEY'): # 检查是否通过 instance config 设置了
    #     raise ValueError("生产环境中未设置 SECRET_KEY")
    # if not Config.JWT_SECRET_KEY and not os.environ.get('JWT_SECRET_KEY'):
    #     raise ValueError("生产环境中未设置 JWT_SECRET_KEY")
    # if Config.SQLALCHEMY_DATABASE_URI.startswith('sqlite'): # 避免生产使用默认的sqlite
    #      if not os.environ.get('DATABASE_URL'): # 检查环境变量是否设置了生产数据库
    #         raise ValueError("生产环境的 DATABASE_URL 未设置或仍为开发默认值")

    # 生产环境特定配置
    SESSION_COOKIE_SECURE = True  # Cookie 只通过 HTTPS 发送
    SESSION_COOKIE_HTTPONLY = True  # 禁止客户端 JavaScript 访问 Cookie
    SESSION_COOKIE_SAMESITE = 'Lax'  # SameSite 属性防止 CSRF
    # PREFERRED_URL_SCHEME = 'https'  # 如果部署在 HTTPS 之后


# 将配置类映射到字典，方便通过名称加载
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig  # 应用默认使用的配置
}


