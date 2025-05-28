#!/usr/bin/env python
"""
重置和重建 Elasticsearch 索引
"""
import os
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from app.utils.es_utils import es_client, IMAGE_INDEX_NAME, NEWS_INDEX_NAME
    from app.utils.es_utils import create_index, create_news_index
except ImportError:
    logger.error("无法导入必要的模块，请确保在正确的目录中运行此脚本")
    sys.exit(1)

def reset_indices():
    """重置并重建所有索引"""
    try:
        # 检查 Elasticsearch 连接
        if not es_client.ping():
            logger.error("无法连接到 Elasticsearch，请确保服务已启动")
            return False
        
        for index_name in [ NEWS_INDEX_NAME]:
            if es_client.indices.exists(index=index_name):
                logger.info(f"删除索引: {index_name}")
                es_client.indices.delete(index=index_name)
            else:
                logger.info(f"索引不存在: {index_name}")

        for index_name in [ IMAGE_INDEX_NAME]:
            if es_client.indices.exists(index=index_name):
                logger.info(f"删除索引: {index_name}")
                es_client.indices.delete(index=index_name)
            else:
                logger.info(f"索引不存在: {index_name}")

        logger.info("创建新闻向量索引...")
        create_news_index()
        create_index()
        
        logger.info("索引重置完成")
        return True
    except Exception as e:
        logger.error(f"重置索引时出错: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("开始重置 Elasticsearch 索引...")
    success = reset_indices()
    if success:
        logger.info("索引重置成功")
    else:
        logger.error("索引重置失败") 