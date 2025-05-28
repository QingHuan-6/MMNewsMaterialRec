#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据库迁移脚本：添加thumbnail_url列到images表
"""

import pymysql
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
    'charset': 'utf8mb4'
}

def migrate_database():
    """执行数据库迁移"""
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
        if conn:
            conn.rollback()
        logger.error(f"数据库迁移出错: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    logger.info("开始数据库迁移...")
    if migrate_database():
        logger.info("数据库迁移成功完成")
    else:
        logger.error("数据库迁移失败") 