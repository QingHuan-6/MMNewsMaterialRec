import pymysql

# 连接数据库
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='123456',
    charset='utf8mb4'
)

try:
    with connection.cursor() as cursor:
        # 创建数据库
        cursor.execute("CREATE DATABASE IF NOT EXISTS pixtock CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.execute("USE pixtock")
        
        # 创建用户表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL,
            email VARCHAR(100) UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """)
        
        # 创建图片表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id VARCHAR(36) PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            file_path VARCHAR(255) NOT NULL,
            url VARCHAR(255) NOT NULL,
            original_url VARCHAR(255) NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_id INT,
            embedding LONGBLOB,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        )
        """)
        
        # 创建标签表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(50) NOT NULL UNIQUE
        )
        """)
        
        # 创建图片标签关联表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_tags (
            image_id VARCHAR(36),
            tag_id INT,
            PRIMARY KEY (image_id, tag_id),
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
        )
        """)
        
        # 创建图片描述表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS captions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_id VARCHAR(36),
            caption TEXT NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
        )
        """)
        
        # 创建收藏表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS favorites (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            image_id VARCHAR(36),
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY (user_id, image_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
        )
        """)
        
        # 创建下载历史表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS downloads (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            image_id VARCHAR(36),
            downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
        )
        """)
        
    # 提交更改
    connection.commit()
    print("数据库和表创建成功！")
    
except Exception as e:
    print(f"发生错误: {e}")
    
finally:
    connection.close()