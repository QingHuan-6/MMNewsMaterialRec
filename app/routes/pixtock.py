from flask import Blueprint, request, jsonify,current_app
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, verify_jwt_in_request
import time
from ..models.models import User, Image, Tag, ImageTag, Caption, Favorite, Download, SearchHistory, ViewHistory, News
from ..models.models import db
import uuid
import os
import torch
import numpy as np
import pickle
from PIL import Image as PILImage
from cn_clip.clip import tokenize
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from . import pixtock_bp

# 导入工具函数
from ..utils.utils import (
    load_cn_clip_model, 
    allowed_file, 
    generate_thumbnail, 
    add_thumbnail_url, 
    load_embeddings_from_db, 
    get_embeddings, 
    update_embedding,
    find_similar_images_by_embedding
)

# 导入 Elasticsearch 工具函数
from ..utils.es_utils import (
    create_index,
    index_image_vector,
    search_similar_vectors,
    search_by_text_embedding,
    bulk_index_vectors
)

device = "cuda" if torch.cuda.is_available() else "cpu"
Server_IP = "localhost:5000"
# 配置参数 - 在原有Flask配置之后添加
# 修改模型和图片文件夹设置
MODEL_NAME = "ViT-B-16"
CNA_FOLDER = 'CNA_images'
CACHE_DIR = "CNA_cache"
IMAGE_EMBED_CACHE = os.path.join(CACHE_DIR, "CNA_image_embeddings.npz")

# 添加缩略图目录配置
THUMBNAIL_FOLDER = 'thumbnails'
THUMBNAIL_SIZE = (200, 200)  # 缩略图尺寸

# 确保目录存在
if not os.path.exists(CNA_FOLDER):
    os.makedirs(CNA_FOLDER)
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
if not os.path.exists(THUMBNAIL_FOLDER):
    os.makedirs(THUMBNAIL_FOLDER)

# 配置路径
FLICKR_FOLDER = 'Flickr8K/Images'
CAPTIONS_FILE = 'Flickr8K/captions.txt'
UPLOAD_FOLDER = 'uploads'

# 初始化cn_clip模型 - 使用工具模块的懒加载函数
cn_clip_model, cn_preprocess = load_cn_clip_model()

# 确保在应用启动时创建索引
create_index()

#通过关键词搜索图片(对应pixtock1.py,但逻辑直接写在此处了)
@pixtock_bp.route('/api/images/search', methods=['GET'])
def search_images():
    """通过关键词搜索图片 (使用 CN-CLIP 和 Elasticsearch)"""
    start_time = time.time()
    current_app.logger.info("开始搜索...")
    
    query = request.args.get('query', '')
    
    if not query:
        return jsonify({'error': '搜索关键词不能为空'}), 400
    
    try:
        verify_jwt_in_request(optional=True)
        user_id = get_jwt_identity()
        if user_id:
            new_search = SearchHistory(
                user_id=user_id,
                search_content=query
            )
            db.session.add(new_search)
            db.session.commit()
    except Exception as e:
        current_app.logger.error(f"记录检索历史时出错: {str(e)}")
    
    try:
        # 获取 CN-CLIP 模型
        cn_clip_model, _ = load_cn_clip_model()
        if not cn_clip_model:
            return jsonify({'error': 'Model not available'}), 500
        
        # 对查询文本进行编码
        text_inputs = tokenize([query]).to(device)
        with torch.no_grad():
            text_embedding = cn_clip_model.encode_text(text_inputs).cpu().numpy()
        
        # 使用 Elasticsearch 搜索相似图片
        search_results = search_by_text_embedding(text_embedding[0])
        
        # 处理搜索结果
        for item in search_results:
            # 添加缩略图 URL
            item = add_thumbnail_url(item, Server_IP)
        
        current_app.logger.info(f"搜索完成，找到 {len(search_results)} 个结果，耗时: {time.time() - start_time:.4f}秒")
        
        return jsonify({
            'total': len(search_results),
            'images': search_results
        })
        
    except Exception as e:
        current_app.logger.error(f"图片搜索出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'搜索处理失败: {str(e)}'}), 500


#获取与给定图片相似的其他图片(对应pixtock2.py,但逻辑直接写在此处了)
@pixtock_bp.route('/api/images/<image_id>/similar', methods=['GET'])
def get_similar_images(image_id):
    """获取与给定图片相似的其他图片"""
    max_results = int(request.args.get('max_results', 20))

    try:
        # 查询数据库获取图片
        image = Image.query.get(image_id)
        if not image:
            return jsonify({'error': 'Image not found'}), 404

        # 获取图片的向量表示
        if not image.embedding:
            return jsonify({'error': 'Image has no embedding'}), 400
        
        embedding = pickle.loads(image.embedding)

        # 使用 Elasticsearch 搜索相似图片
        similar_images = search_similar_vectors(
            query_vector=embedding,
            top_k=max_results,
            exclude_id=image_id
        )
        for item in similar_images:
            item["score"] = item["score"] -1
        
        # 记录用户浏览历史
        try:
            verify_jwt_in_request(optional=True)
            user_id = get_jwt_identity()
            if user_id:
                existing_view = ViewHistory.query.filter_by(user_id=user_id, image_id=image_id).first()
                if existing_view:
                    db.session.delete(existing_view)

                new_view = ViewHistory(
                    user_id=user_id,
                    image_id=image_id
                )
                db.session.add(new_view)
                db.session.commit()
        except Exception as e:
            current_app.logger.error(f"记录浏览历史时出错: {e}")

        return jsonify({
            'image_id': image_id,
            'similar_images': similar_images
        })

    except Exception as e:
        current_app.logger.error(f"获取相似图片时出错: {e}")
        return jsonify({'error': str(e)}), 500


#获取用户的个性化推荐(使用RecSys.py中的个性化推荐算法)
@pixtock_bp.route('/api/personalized_recommendations', methods=['GET', 'OPTIONS'])
def personalized_recommendations():
    """获取用户的个性化推荐"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        return response

    # 验证用户登录
    try:
        verify_jwt_in_request()
        user_id = get_jwt_identity()
        if not user_id:
            return jsonify({'error': '需要用户登录'}), 401
    except Exception as e:
        return jsonify({'error': f'验证用户身份失败: {str(e)}'}), 401

    # 获取请求参数
    limit = request.args.get('limit', 120, type=int)
    refresh = request.args.get('refresh', 'false').lower() == 'true'

    try:
        # 导入并使用推荐系统
        from ..services.RecSys import get_recommender_instance

        # 获取推荐系统实例
        recommender = get_recommender_instance(current_app)

        # 生成推荐
        start_time = time.time()
        recommendations = recommender.get_recommendation_details(
            user_id=user_id,
            top_n=limit,
            use_cache=False
        )

        # 记录耗时
        elapsed_time = time.time() - start_time
        print(f"为用户 {user_id} 生成推荐耗时 {elapsed_time:.2f} 秒")

        # 返回结果
        return jsonify({
            'total': len(recommendations),
            'recommendations': recommendations,
            'elapsed_time': elapsed_time
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'获取个性化推荐失败: {str(e)}'}), 500

# 添加一个新的路由用于将现有向量数据迁移到 Elasticsearch
@pixtock_bp.route('/api/admin/migrate_vectors', methods=['POST'])
def migrate_vectors_to_es():
    """将现有的向量数据迁移到 Elasticsearch"""
    try:
        # 获取所有图片数据
        images = Image.query.all()
        vectors_data = []

        for image in images:
            if image.embedding:
                vector = pickle.loads(image.embedding)
                metadata = {
                    'image_id': image.id,
                    'url': image.url,
                    'title': image.title,
                    'file_path': image.file_path,
                    'captions': [c.caption for c in image.captions],
                    'thumbnail_url': image.thumbnail_url
                }
                vectors_data.append({
                    'image_id': image.id,
                    'vector': vector,
                    **metadata
                })

        # 批量索引向量数据
        bulk_index_vectors(vectors_data)

        return jsonify({
            'message': f'成功迁移 {len(vectors_data)} 条向量数据到 Elasticsearch',
            'count': len(vectors_data)
        })

    except Exception as e:
        current_app.logger.error(f"迁移向量数据时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500
