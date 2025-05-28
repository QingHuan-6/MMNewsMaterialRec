#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
为数据库中的Image表添加缩略图URL字段并填充数据
"""

import os
import pymysql
import hashlib
from PIL import Image as PILImage
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'pixtock',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# 缩略图配置
THUMBNAIL_FOLDER = 'thumbnails'
THUMBNAIL_SIZE = (200, 200)
SERVER_IP = 'localhost:5000'

def generate_thumbnail(image_path, thumbnail_folder=THUMBNAIL_FOLDER, max_size=THUMBNAIL_SIZE):
    """
    生成图片的缩略图并返回缩略图路径
    
    参数:
        image_path: 原图路径
        thumbnail_folder: 缩略图存储文件夹路径
        max_size: 缩略图最大尺寸，默认为(200, 200)
    
    返回:
        str: 缩略图路径
    """
    try:
        # 确保缩略图目录存在
        if not os.path.exists(thumbnail_folder):
            os.makedirs(thumbnail_folder)
            
        # 计算缩略图文件名
        image_name = os.path.basename(image_path)
        file_hash = hashlib.md5(image_path.encode()).hexdigest()
        thumbnail_name = f"{file_hash}_{max_size[0]}x{max_size[1]}.jpg"
        thumbnail_path = os.path.join(thumbnail_folder, thumbnail_name)
        
        # 如果缩略图已存在，直接返回路径
        if os.path.exists(thumbnail_path):
            return thumbnail_path
        
        # 检查原图是否存在
        if not os.path.exists(image_path):
            logger.warning(f"原图不存在: {image_path}")
            return None
            
        # 打开原图
        img = PILImage.open(image_path).convert('RGB')
        
        # 创建缩略图
        img.thumbnail(max_size, PILImage.LANCZOS)
        
        # 保存缩略图
        img.save(thumbnail_path, "JPEG", quality=85)
        
        return thumbnail_path
    except Exception as e:
        logger.error(f"生成缩略图失败: {e}")
        return None

def add_thumbnail_url_column():
    """添加缩略图URL列到数据库"""
    conn = None
    try:
        # 连接到数据库
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 检查thumbnail_url列是否已存在
        cursor.execute("SHOW COLUMNS FROM images LIKE 'thumbnail_url'")
        column_exists = cursor.fetchone()
        
        # 如果列不存在，则添加
        if not column_exists:
            logger.info("添加thumbnail_url列到images表")
            cursor.execute("ALTER TABLE images ADD COLUMN thumbnail_url VARCHAR(255)")
            conn.commit()
            logger.info("成功添加thumbnail_url列")
        else:
            logger.info("thumbnail_url列已存在")
        
        return True
    except Exception as e:
        logger.error(f"添加缩略图URL列时出错: {e}")
        return False
    finally:
        if conn:
            conn.close()

def update_thumbnail_urls():
    """更新所有图片的缩略图URL"""
    conn = None
    try:
        # 连接到数据库
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 获取所有图片
        cursor.execute("SELECT id, file_path, url FROM images")
        images = cursor.fetchall()
        
        logger.info(f"找到 {len(images)} 张图片需要更新缩略图URL")
        
        # 更新计数器
        updated_count = 0
        skipped_count = 0
        
        # 处理每张图片
        for img in images:
            image_id = img['id']
            file_path = img['file_path']
            url = img['url']
            
            thumbnail_url = None
            
            # 如果有文件路径，生成缩略图
            if file_path and os.path.exists(file_path):
                thumbnail_path = generate_thumbnail(file_path)
                if thumbnail_path:
                    thumbnail_url = f"http://{SERVER_IP}/thumbnails/{os.path.basename(thumbnail_path)}"
            
            # 如果没有生成缩略图，使用原图URL
            if not thumbnail_url and url:
                thumbnail_url = url
            
            # 更新数据库
            if thumbnail_url:
                cursor.execute(
                    "UPDATE images SET thumbnail_url = %s WHERE id = %s",
                    (thumbnail_url, image_id)
                )
                updated_count += 1
            else:
                skipped_count += 1
                
            # 每100张图片提交一次
            if updated_count % 100 == 0:
                conn.commit()
                logger.info(f"已更新 {updated_count} 张图片的缩略图URL")
        
        # 最终提交
        conn.commit()
        logger.info(f"缩略图URL更新完成: 更新 {updated_count} 张，跳过 {skipped_count} 张")
        
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"更新缩略图URL时出错: {e}")
        return False
    finally:
        if conn:
            conn.close()

def main():
    """主函数"""
    logger.info("开始处理缩略图URL...")
    
    # 添加缩略图URL列
    if add_thumbnail_url_column():
        # 更新缩略图URL
        update_thumbnail_urls()
    
    logger.info("处理完成")

if __name__ == "__main__":
    main() 