from app import create_app
import logging

if __name__ == '__main__':
    # 设置日志级别为INFO，以便查看蓝图注册信息
    logging.basicConfig(level=logging.INFO)
    
    app = create_app(config_name='development')
    print(f"应用将在 http://{app.config['SERVER_IP']} 上启动")
    app.run(debug=True, host='0.0.0.0', use_reloader=False)  # 禁用自动重载功能