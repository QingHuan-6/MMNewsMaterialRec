from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import numpy as np
import pickle
from typing import List, Dict, Any
import logging
# Elasticsearch 配置
ES_HOST = "127.0.0.1"
ES_PORT = 9200
IMAGE_INDEX_NAME = "image_vectors"
NEWS_INDEX_NAME = "news_vectors"
logger = logging.getLogger(__name__)
# --- 将您的 Elasticsearch 密码直接写在这里 ---
# 注意：这种方式仅推荐在本地开发和测试环境中使用。
# 在生产环境中，绝对不要将敏感信息（如密码）硬编码在代码中！
# 生产环境应使用环境变量、密钥管理服务（如Vault）或配置文件等更安全的机制。
ELASTIC_USERNAME = "elastic"
ELASTIC_PASSWORD = "WsxeY2XNiNL10A1kLoRs" # <--- 将您获取到的 elastic 密码粘贴到这里

# 创建 Elasticsearch 客户端
try:
    es_client = Elasticsearch(
        hosts=["https://127.0.0.1:9200"],
        basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD),
        verify_certs=False
    )
    if es_client.ping():
        logger.info("Successfully connected to Elasticsearch")
    else:
        logger.error("Elasticsearch ping failed")
        raise ConnectionError("无法连接到 Elasticsearch")
except Exception as e:
    logger.error(f"连接 Elasticsearch 失败: {e}")
    raise

def create_index():
    """创建图片向量索引"""
    mapping = {
        "mappings": {
            "properties": {
                "image_id": {"type": "keyword"},
                "vector": {
                    "type": "dense_vector",
                    "dims": 512,  # CN-CLIP 向量维度
                    "index": True,
                    "similarity": "cosine"
                },
                "url": {"type": "keyword"},
                "title": {"type": "text"},
                "file_path": {"type": "keyword"},
                "captions": {"type": "text"},
                "thumbnail_url": {"type": "keyword"}
            }
        },
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
    }
    
    if not es_client.indices.exists(index=IMAGE_INDEX_NAME):
        es_client.indices.create(index=IMAGE_INDEX_NAME, body=mapping)
        logger.info(f"Created index: {IMAGE_INDEX_NAME}")
    
    # 创建新闻向量索引
    #create_news_index()

def create_news_index():
    """创建新闻向量索引"""
    mapping = {
        "mappings": {
            "properties": {
                "image_id": {"type": "keyword"},  # 兼容批量索引函数使用的字段名
                "news_id": {"type": "keyword"},
                "vector": {
                    "type": "dense_vector",
                    "dims": 384,  # 向量维度
                    "index": True,
                    "similarity": "cosine"
                },
                "title": {"type": "text", "analyzer": "ik_max_word"},
                "content": {"type": "text", "analyzer": "ik_max_word"},
                "topic": {"type": "keyword"},
                "sentiment": {"type": "keyword"}
            }
        },
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "ik_max_word": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase"]
                        }
                    }
                }
            }
        }
    }
    
    try:
        if not es_client.indices.exists(index=NEWS_INDEX_NAME):
            es_client.indices.create(index=NEWS_INDEX_NAME, body=mapping)
            logger.info(f"Created index: {NEWS_INDEX_NAME}")
        else:
            logger.info(f"Index {NEWS_INDEX_NAME} already exists")
    except Exception as e:
        logger.error(f"创建新闻索引失败: {str(e)}")
        # 尝试使用更简单的映射
        try:
            simple_mapping = {
                "mappings": {
                    "properties": {
                        "image_id": {"type": "keyword"},
                        "news_id": {"type": "keyword"},
                        "vector": {
                            "type": "dense_vector",
                            "dims": 512,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "title": {"type": "text"},
                        "content": {"type": "text"},
                        "topic": {"type": "keyword"},
                        "sentiment": {"type": "keyword"}
                    }
                },
                "settings": {
                    "index": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                }
            }
            
            # 如果索引已存在，先删除
            if es_client.indices.exists(index=NEWS_INDEX_NAME):
                es_client.indices.delete(index=NEWS_INDEX_NAME)
                logger.info(f"Deleted existing index: {NEWS_INDEX_NAME}")
            
            es_client.indices.create(index=NEWS_INDEX_NAME, body=simple_mapping)
            logger.info(f"Created index with simple mapping: {NEWS_INDEX_NAME}")
        except Exception as e2:
            logger.error(f"创建简化新闻索引也失败: {str(e2)}")

def index_image_vector(image_id: str, vector: np.ndarray, metadata: Dict[str, Any]):
    """索引单个图片向量"""
    vector_data = {
        "image_id": image_id,
        "vector": vector.tolist(),
        **metadata
    }
    # 根据ID前缀判断是图片还是新闻
    if str(image_id).startswith("news_"):
        es_client.index(index=NEWS_INDEX_NAME, id=image_id, document=vector_data)
    else:
        es_client.index(index=IMAGE_INDEX_NAME, id=image_id, document=vector_data)

def index_news_vector(news_id: str, vector: np.ndarray, metadata: Dict[str, Any]):
    """索引单个新闻向量"""
    vector_data = {
        "news_id": news_id,
        "vector": vector.tolist(),
        **metadata
    }
    es_client.index(index=NEWS_INDEX_NAME, id=news_id, document=vector_data)

def bulk_index_vectors(vectors_data: List[Dict[str, Any]]):
    """批量索引图片和新闻向量"""
    image_actions = []
    news_actions = []
    
    # 记录处理的数据
    logger.info(f"准备批量索引 {len(vectors_data)} 条向量数据")
    
    for data in vectors_data:
        try:
            image_id = data["image_id"]
            
            # 确保向量数据是列表类型
            if "vector" in data:
                if isinstance(data["vector"], np.ndarray):
                    vector_list = data["vector"].tolist()
                elif isinstance(data["vector"], list):
                    vector_list = data["vector"]
                else:
                    logger.warning(f"ID为 {image_id} 的向量数据类型不正确: {type(data['vector'])}")
                    continue
            else:
                logger.warning(f"ID为 {image_id} 的数据缺少向量字段")
                continue
            
            # 准备源数据
            vector_source = {
                "vector": vector_list,
                **{k: v for k, v in data.items() if k != "vector"}
            }
            
            # 根据ID前缀判断是图片还是新闻
            if str(image_id).startswith("news_"):
                news_actions.append({
                    "_index": NEWS_INDEX_NAME,
                    "_id": image_id,
                    "_source": vector_source
                })
            else:
                image_actions.append({
                    "_index": IMAGE_INDEX_NAME,
                    "_id": image_id,
                    "_source": vector_source
                })
        except Exception as e:
            logger.error(f"处理向量数据时出错: {str(e)}")
            continue
    
    # 批量索引图片向量
    if image_actions:
        try:
            # 分批处理，每批最多100条
            batch_size = 100
            for i in range(0, len(image_actions), batch_size):
                batch = image_actions[i:i+batch_size]
                success, failed = bulk(es_client, batch, stats_only=False, raise_on_error=False)
                logger.info(f"图片向量批量索引: 成功 {success} 条, 失败 {len(failed) if failed else 0} 条")
                if failed:
                    for item in failed:
                        logger.error(f"图片向量索引失败: {item}")
        except Exception as e:
            logger.error(f"批量索引图片向量时出错: {str(e)}")
    
    # 批量索引新闻向量
    if news_actions:
        try:
            # 分批处理，每批最多100条
            batch_size = 100
            for i in range(0, len(news_actions), batch_size):
                batch = news_actions[i:i+batch_size]
                success, failed = bulk(es_client, batch, stats_only=False, raise_on_error=False)
                logger.info(f"新闻向量批量索引: 成功 {success} 条, 失败 {len(failed) if failed else 0} 条")
                if failed:
                    for item in failed:
                        logger.error(f"新闻向量索引失败: {item}")
        except Exception as e:
            logger.error(f"批量索引新闻向量时出错: {str(e)}")

def search_similar_vectors(query_vector: np.ndarray, top_k: int = 10, exclude_id: str = None, index_name: str = None) -> List[Dict[str, Any]]:
    """
    搜索相似向量
    
    参数:
        query_vector: 查询向量
        top_k: 返回的结果数量
        exclude_id: 要排除的ID
        index_name: 要搜索的索引名称，如果为None则搜索所有索引
    """
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                "params": {"query_vector": query_vector.tolist()}
            }
        }
    }
    
    # 确定要搜索的索引
    if index_name is None:
        # 默认搜索所有索引
        indices = f"{IMAGE_INDEX_NAME},{NEWS_INDEX_NAME}"
    else:
        indices = index_name
    
    response = es_client.search(
        index=indices,
        body={
            "query": script_query,
            "size": top_k + (1 if exclude_id else 0)
        }
    )
    
    results = []
    for hit in response["hits"]["hits"]:
        if exclude_id and hit["_id"] == exclude_id:
            continue
        result = {
            "id": hit["_id"],
            "score": hit["_score"],
            **{k: v for k, v in hit["_source"].items() if k != "vector"}
        }
        results.append(result)
    
    return results[:top_k]

def search_by_text_embedding(text_embedding: np.ndarray, top_k: int = 150, index_name: str = None) -> List[Dict[str, Any]]:
    return search_similar_vectors(text_embedding, top_k, index_name=index_name)

def search_similar_news(query_vector: np.ndarray, top_k: int = 10, exclude_id: str = None) -> List[Dict[str, Any]]:
    """搜索相似新闻"""
    return search_similar_vectors(query_vector, top_k, exclude_id, index_name=NEWS_INDEX_NAME)

def search_similar_images(query_vector: np.ndarray, top_k: int = 10, exclude_id: str = None) -> List[Dict[str, Any]]:
    """搜索相似图片"""
    return search_similar_vectors(query_vector, top_k, exclude_id, index_name=IMAGE_INDEX_NAME) 



if __name__ == "__main__":
    create_index()