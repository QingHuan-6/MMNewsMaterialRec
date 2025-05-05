import os
import sys
from werkzeug.security import generate_password_hash
from flask import Flask

# 确保能导入models模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import db, User

# 创建Flask应用
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost/pixtock'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def create_admin_user(username='admin', password='admin', email='admin@example.com'):
    """
    创建管理员用户
    
    Args:
        username: 管理员用户名，默认为'admin'
        password: 管理员密码，默认为'admin'
        email: 管理员邮箱，默认为'admin@example.com'
    """
    with app.app_context():
        # 检查用户是否已存在
        existing_user = User.query.filter_by(username=username).first()
        
        if existing_user:
            # 更新为管理员权限
            existing_user.role = 'admin'
            db.session.commit()
            print(f"用户 {username} 已存在，已更新为管理员角色")
        else:
            # 创建新管理员用户
            hashed_password = generate_password_hash(password)
            admin_user = User(
                username=username,
                password=hashed_password,
                email=email,
                role='admin'
            )
            db.session.add(admin_user)
            db.session.commit()
            print(f"已成功创建管理员用户 {username}")

if __name__ == '__main__':
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='创建管理员用户')
    parser.add_argument('--username', default='admin', help='管理员用户名')
    parser.add_argument('--password', default='admin', help='管理员密码')
    parser.add_argument('--email', default='admin@example.com', help='管理员邮箱')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        # 创建管理员用户
        create_admin_user(args.username, args.password, args.email)
    except Exception as e:
        print(f"创建管理员失败: {str(e)}")
        sys.exit(1) 