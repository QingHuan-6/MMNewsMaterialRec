# backend/migrate_role.py
import os
import sys
from flask import Flask
from sqlalchemy import text
# 确保能导入models模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import db, User

# 创建Flask应用
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost/pixtock'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    try:
        # 执行SQL语句直接修改表结构
        db.session.execute(text('ALTER TABLE users ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT "user"'))
        db.session.commit()
        print("已成功添加role字段到users表")
    except Exception as e:
        print(f"添加字段失败: {e}")
        # 检查是否因为字段已存在导致的错误
        if "Duplicate column name" in str(e):
            print("role字段已存在，无需添加")
        else:
            raise