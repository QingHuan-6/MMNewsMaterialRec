# backend/app/__init__.py
import os
import logging # Python 内置的日志模块
import warnings # Python 内置的警告模块
import torch # PyTorch 深度学习框架
import numpy as np
import pickle
import uuid
import time
from datetime import datetime
from PIL import Image as PILImage
import hashlib

from flask import Flask, request, jsonify, send_from_directory # Flask Web 框架
from flask_jwt_extended import JWTManager # 处理 JWT (JSON Web Tokens) 的 Flask 扩展
from flask_cors import CORS # 处理跨域资源共享 (CORS) 的 Flask 扩展
from transformers import logging as transformers_logging # Hugging Face Transformers 库的日志模块
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity




# 从项目根目录的 config.py 导入配置字典
# 使用 as 重命名以避免与 Python 内置的 config 模块潜在冲突
from .config import config as app_configs
# 从同级目录的 models.py 导入 db 实例和模型类
from .models.models import db, User, Image, Tag, ImageTag, Caption, Favorite, Download, SearchHistory, ViewHistory, News



# 应用级的模型实例变量 - 将在 create_app 中初始化
device_instance = "cpu" # 默认为 CPU
news_analyzer_instance = None
image_recommender_instance = None

# 图片嵌入相关变量
image_files_list = []
image_embeddings_array = np.zeros((0, 512), dtype=np.float32)
image_captions = {}

# 尽早抑制警告信息
warnings.filterwarnings("ignore") # 忽略所有普通警告
transformers_logging.set_verbosity_error() # Transformers 库只报告错误
logging.getLogger("PIL").setLevel(logging.ERROR) # Pillow (PIL) 库只报告错误
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制 TensorFlow C++ 后端的日志 (如果使用了 TensorFlow)


def create_app(config_name=None):
    """应用工厂函数，用于创建和配置 Flask 应用实例。"""
    if config_name is None:
        # 如果未提供配置名称，尝试从操作系统环境变量 FLASK_CONFIG 读取
        # 如果环境变量也未设置，则使用 'default' 配置
        config_name = os.environ.get('FLASK_CONFIG', 'default')

    # 创建 Flask 应用实例
    # instance_relative_config=True 允许从 instance/ 文件夹加载配置
    app = Flask(__name__, instance_relative_config=True)

    # 1. 从 config.py 中的配置对象加载配置
    app.config.from_object(app_configs[config_name])
    # 2. (可选) 调用配置类中定义的 init_app 方法 (如果存在)
    if hasattr(app_configs[config_name], 'init_app'):
        app_configs[config_name].init_app(app)

    # 3. 从 instance/config.py 加载配置 (如果存在)
    # 这个文件用于存放敏感信息，如生产环境的密钥，不应提交到版本库
    # 它会覆盖之前加载的同名配置项
    # silent=True 表示如果 instance/config.py 文件不存在，则不报错
    app.config.from_pyfile('config.py', silent=True)

    # 配置应用日志级别
    app.logger.setLevel(app.config.get('LOG_LEVEL', logging.ERROR)) # 从配置获取或默认为ERROR

    # 初始化 Flask 扩展
    db.init_app(app) # 初始化 SQLAlchemy
    CORS(app) # 初始化 CORS，可以接受 app.config 中的 CORS 相关配置
    jwt = JWTManager(app) # 初始化 JWT管理器

    # 确定运行设备 (CPU 或 CUDA)
    global device_instance # 声明我们要修改全局变量
    device_instance = "cuda" if torch.cuda.is_available() else "cpu"
    app.device = device_instance # 将设备信息附加到 app 对象上，方便其他地方访问
    app.logger.info(f"应用将使用设备: {app.device}")



    # 在应用上下文中执行操作
    with app.app_context():
        # 注册Blueprint
        from .routes import news_bp, images_bp,novo_bp, pixtock_bp, set_weights_bp, user_bp
        app.register_blueprint(news_bp)
        app.register_blueprint(images_bp)
        app.register_blueprint(novo_bp)
        app.register_blueprint(pixtock_bp)
        app.register_blueprint(set_weights_bp)
        app.register_blueprint(user_bp)


        # 创建管理员账户
        try:
            # 检查admin账户是否存在
            from werkzeug.security import generate_password_hash
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                # 如果不存在，创建admin账户
                hashed_password = generate_password_hash('admin')
                admin_user = User(
                    username='admin',
                    password=hashed_password,
                    email='admin@example.com',
                    role='admin',
                    is_admin=True
                )
                db.session.add(admin_user)
                db.session.commit()
                app.logger.info("已创建管理员账户，用户名: admin，密码: admin")
            elif admin.role != 'admin' or not admin.is_admin:
                # 如果存在但不是管理员，更新为管理员
                admin.role = 'admin'
                admin.is_admin = True
                db.session.commit()
                app.logger.info("已将用户 admin 更新为管理员角色")
            else:
                app.logger.info("管理员账户已存在")
        except Exception as e:
            app.logger.error(f"创建管理员账户时出错: {e}")
            db.session.rollback()
            
        # 迁移图片URL (如果需要)
        try:
            migrate_image_urls(app)
        except Exception as e:
            app.logger.error(f"迁移图片URL时出错: {e}")
        

        app.logger.info("Flask 应用已创建并配置完成。")

    return app


def migrate_image_urls(app):
    """迁移数据库中的图片URL，将localhost:5000替换为当前Server_IP"""
    try:
        app.logger.info(f"开始迁移数据库中的图片URL...")
        import re
        IP_PORT_REGEX = re.compile(
            r'https?:\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+\/',  # 匹配 `http://IP:PORT/`
            flags=re.IGNORECASE  # 忽略大小写
        )
        images = Image.query.all()
        update_count = 0
        
        for img in images:
            # 替换 img.url
            if IP_PORT_REGEX.search(img.url):
                img.url = IP_PORT_REGEX.sub(f'http://{app.config["SERVER_IP"]}/', img.url)  # 替换为 SERVER_IP
                update_count += 1
            
            # 替换 img.original_url
            if IP_PORT_REGEX.search(img.original_url):
                img.original_url = IP_PORT_REGEX.sub(f'http://{app.config["SERVER_IP"]}/', img.original_url)
                
            if IP_PORT_REGEX.search(img.thumbnail_url):
                img.thumbnail_url = IP_PORT_REGEX.sub(f'http://{app.config["SERVER_IP"]}/', img.thumbnail_url)
        if update_count > 0:
            db.session.commit()
            app.logger.info(f"成功更新 {update_count} 条图片记录的URL")
        else:
            app.logger.info("没有发现需要更新的URL")
    except Exception as e:
        app.logger.error(f"迁移图片URL时出错: {e}")
        db.session.rollback()