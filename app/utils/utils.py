"""
工具模块：提供应用程序需要的工具函数和共享资源
特别是实现CLIP模型的懒加载单例模式，避免多次加载相同模型
"""

import os
import torch
import hashlib
from PIL import Image as PILImage
import numpy as np
import pickle
import time
from flask import current_app

# 全局CLIP模型实例 - 初始为None，首次调用时才会加载
_cn_clip_model = None
_cn_clip_preprocess = None

# 全局图片嵌入缓存
_image_files_list = []
_image_embeddings_array = np.zeros((0, 512), dtype=np.float32)
_image_captions = {}

def load_cn_clip_model():
    """
    懒加载CN-CLIP模型 - 只有在首次调用时才会加载模型
    返回: (model, preprocess_fn) 元组
    """
    global _cn_clip_model, _cn_clip_preprocess
    
    # 如果模型已经加载，则直接返回
    if _cn_clip_model is not None and _cn_clip_preprocess is not None:
        return _cn_clip_model, _cn_clip_preprocess
    
    # 首次加载模型
    try:
        from cn_clip.clip import load_from_name
        
        # 确定运行设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 获取模型名称（从配置或使用默认值）
        model_name = current_app.config.get('CLIP_MODEL_NAME', 'ViT-B-16')
        
        # 加载模型
        model, preprocess = load_from_name(model_name, device=device)
        model.eval()  # 设置为评估模式
        
        # 保存到全局变量
        _cn_clip_model = model
        _cn_clip_preprocess = preprocess
        
        current_app.logger.info(f"CN-CLIP模型({model_name})加载成功")
        return model, preprocess
    except Exception as e:
        current_app.logger.error(f"CN-CLIP模型加载失败: {e}")
        return None, None

def load_embeddings_from_db():
    """
    从数据库加载图片嵌入向量和描述
    在应用上下文中运行，共享给不同模块使用
    
    返回: (image_files_list, image_embeddings_array, image_captions) 元组
    """
    global _image_files_list, _image_embeddings_array, _image_captions
    
    try:
        # 导入数据库模型
        from ..models.models import Image
        
        # 从数据库获取所有图片及其嵌入向量
        all_images = Image.query.all()
        image_files_list = []
        embeddings_list = []
        image_captions = {}  # 重置图片描述字典
        
        # 记录处理开始
        current_app.logger.info("开始从数据库加载图片嵌入向量...")
        start_time = time.time()
        
        for img in all_images:
            if img.embedding:
                try:
                    embedding = pickle.loads(img.embedding)
                    image_files_list.append(img.id + '.jpg')  # 假设所有图片都是jpg格式
                    embeddings_list.append(embedding)
                    
                    # 如果有描述，也加载到缓存中
                    if img.captions:
                        caption_texts = [c.caption for c in img.captions]
                        image_captions[img.id + '.jpg'] = caption_texts
                except Exception as e:
                    current_app.logger.error(f"处理图片 {img.id} 的嵌入向量时出错: {e}")
                    continue
        
        # 转换为numpy数组
        image_embeddings_array = np.array(embeddings_list)
        
        # 更新全局变量
        _image_files_list = image_files_list
        _image_embeddings_array = image_embeddings_array
        _image_captions = image_captions
        
        # 记录处理时间
        elapsed_time = time.time() - start_time
        current_app.logger.info(f"已成功加载 {len(image_files_list)} 张图片的嵌入向量，耗时 {elapsed_time:.2f} 秒")
        
        return image_files_list, image_embeddings_array, image_captions
    except Exception as e:
        current_app.logger.error(f"从数据库加载图片嵌入失败: {e}")
        # 返回空结果，以防加载失败
        return [], np.zeros((0, 512), dtype=np.float32), {}

def get_embeddings():
    """
    获取全局缓存的嵌入向量数据，如果不存在则从数据库加载
    
    返回: (image_files_list, image_embeddings_array, image_captions) 元组
    """
    global _image_files_list, _image_embeddings_array, _image_captions
    
    # 如果缓存为空，尝试从数据库加载
    if len(_image_files_list) == 0 or _image_embeddings_array.shape[0] == 0:
        return load_embeddings_from_db()
    
    # 返回缓存的数据
    return _image_files_list, _image_embeddings_array, _image_captions

def update_embedding(image_id, embedding):
    """
    更新缓存中特定图片的嵌入向量（上传新图片后调用）
    
    参数:
        image_id: 图片ID
        embedding: 新的嵌入向量
    
    返回:
        bool: 更新是否成功
    """
    global _image_files_list, _image_embeddings_array
    
    try:
        # 构建文件名格式
        image_file = image_id + '.jpg'
        
        # 检查图片是否已在列表中
        if image_file in _image_files_list:
            # 已存在，更新嵌入向量
            idx = _image_files_list.index(image_file)
            _image_embeddings_array[idx] = embedding
        else:
            # 不存在，添加到列表和数组中
            _image_files_list.append(image_file)
            embedding_reshaped = np.array([embedding])
            if _image_embeddings_array.shape[0] == 0:
                _image_embeddings_array = embedding_reshaped
            else:
                _image_embeddings_array = np.vstack((_image_embeddings_array, embedding_reshaped))
        
        return True
    except Exception as e:
        current_app.logger.error(f"更新图片 {image_id} 嵌入向量时出错: {e}")
        return False

def find_similar_images_by_embedding(embedding, exclude_id=None, max_results=20):
    """
    根据嵌入向量查找相似图片（使用全局缓存或从数据库加载）
    
    参数:
        embedding: 查询嵌入向量
        exclude_id: 要排除的图片ID（可选）
        max_results: 返回的最大结果数
    
    返回:
        list: 相似图片信息的列表
    """
    try:
        from ..models.models import Image
        
        # 首先尝试使用全局缓存
        image_files_list, image_embeddings_array, _ = get_embeddings()
        
        if len(image_files_list) == 0 or image_embeddings_array.shape[0] == 0:
            current_app.logger.warning("嵌入向量缓存为空，将改用数据库查询")
            return find_similar_images_from_db(embedding, exclude_id, max_results)
        
        # 使用向量相似度搜索
        # 归一化查询向量
        embedding_normalized = embedding / np.linalg.norm(embedding)
        
        # 归一化所有嵌入向量
        try:
            embeddings_normalized = image_embeddings_array / np.linalg.norm(image_embeddings_array, axis=1, keepdims=True)
            # 计算余弦相似度
            similarities = np.dot(embeddings_normalized, embedding_normalized)
        except Exception as e:
            current_app.logger.error(f"计算向量相似度时出错: {e}")
            return []
        
        # 获取排序后的索引（相似度从高到低）
        top_indices = np.argsort(-similarities)
        
        # 构建结果列表
        similar_images = []
        count = 0
        
        for idx in top_indices:
            # 从文件名提取图片ID
            img_file = image_files_list[idx]
            img_id = img_file.split('.')[0] if '.' in img_file else img_file
            
            # 如果设置了排除ID且当前ID需要排除，则跳过
            if exclude_id and img_id == exclude_id:
                continue
                
            # 从数据库获取完整图片信息
            img = Image.query.get(img_id)
            if img:
                score = float(similarities[idx])
                
                # 构建图片信息
                image_info = {
                    'id': img.id,
                    'url': img.url,
                    'original_url': img.original_url,
                    'title': img.title,
                    'captions': [c.caption for c in img.captions] if img.captions else [],
                    'tags': [t.name for t in img.tags] if img.tags else [],
                    'score': score,
                    'file_path': img.file_path
                }
                
                # 添加缩略图URL
                image_info = add_thumbnail_url(image_info)
                similar_images.append(image_info)
                count += 1
                
                # 达到最大结果数时停止
                if count >= max_results:
                    break
        
        return similar_images
    except Exception as e:
        current_app.logger.error(f"查找相似图片时出错: {e}")
        return []

def find_similar_images_from_db(embedding, exclude_id=None, max_results=20):
    """
    直接从数据库查找相似图片（当缓存不可用时使用）
    
    参数:
        embedding: 查询嵌入向量
        exclude_id: 要排除的图片ID（可选）
        max_results: 返回的最大结果数
    
    返回:
        list: 相似图片信息的列表
    """
    try:
        from ..models.models import Image
        
        # 获取所有图片
        all_images = Image.query.all()
        
        # 提取图片和嵌入向量
        images_with_embeddings = []
        for img in all_images:
            if img.id != exclude_id and img.embedding:
                try:
                    img_embedding = pickle.loads(img.embedding)
                    images_with_embeddings.append((img, img_embedding))
                except Exception as e:
                    current_app.logger.error(f"处理图片 {img.id} 的嵌入向量时出错: {e}")
                    continue
        
        if not images_with_embeddings:
            return []
        
        # 构建嵌入向量数组
        images = [item[0] for item in images_with_embeddings]
        embeddings = np.array([item[1] for item in images_with_embeddings])
        
        # 使用矩阵运算计算相似度
        embedding_normalized = embedding / np.linalg.norm(embedding)
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = np.dot(embeddings_normalized, embedding_normalized)
        
        # 获取排序后的索引
        top_indices = np.argsort(-similarities)[:max_results]
        
        # 构建响应
        similar_images = []
        for idx in top_indices:
            img = images[idx]
            score = float(similarities[idx])
            
            image_info = {
                'id': img.id,
                'url': img.url,
                'original_url': img.original_url,
                'title': img.title,
                'captions': [c.caption for c in img.captions] if img.captions else [],
                'tags': [t.name for t in img.tags] if img.tags else [],
                'score': score,
                'file_path': img.file_path
            }
            
            # 添加缩略图URL
            image_info = add_thumbnail_url(image_info)
            similar_images.append(image_info)
        
        return similar_images
    except Exception as e:
        current_app.logger.error(f"从数据库查找相似图片时出错: {e}")
        return []

# 通用工具函数

def allowed_file(filename):
    """检查文件名是否是允许的扩展名"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_thumbnail(image_path, thumbnail_folder=None, max_size=(200, 200)):
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
        # 如果没有指定缩略图文件夹，使用应用配置
        if thumbnail_folder is None:
            thumbnail_folder = current_app.config.get('THUMBNAIL_FOLDER')
            
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
        
        # 打开原图
        img = PILImage.open(image_path).convert('RGB')
        
        # 创建缩略图
        img.thumbnail(max_size, PILImage.LANCZOS)
        
        # 保存缩略图
        img.save(thumbnail_path, "JPEG", quality=85)
        
        return thumbnail_path
    except Exception as e:
        current_app.logger.error(f"生成缩略图失败: {e}")
        return None

def add_thumbnail_url(image_dict, server_ip=None):
    """
    给图片数据添加缩略图URL
    
    参数:
        image_dict: 图片数据字典
        server_ip: 服务器IP地址，如果不提供则从配置获取
        
    返回:
        dict: 添加了thumbnail_url的图片字典
    """
    if server_ip is None:
        server_ip = current_app.config.get('SERVER_IP', 'localhost:5000')
        
    if 'file_path' in image_dict and image_dict['file_path']:
        # 对于有文件路径的图片
        thumbnail_path = generate_thumbnail(image_dict['file_path'])
        if thumbnail_path:
            image_dict['thumbnail_url'] = f"http://{server_ip}/thumbnails/{os.path.basename(thumbnail_path)}"
    elif 'url' in image_dict and image_dict['url']:
        # 尝试从URL提取文件路径 (如果是本地文件)
        url_parts = image_dict['url'].split('/')
        if len(url_parts) > 2:
            filename = url_parts[-1]
            folder = url_parts[-2]
            upload_folder = current_app.config.get('UPLOAD_FOLDER')
            cna_folder = current_app.config.get('CNA_FOLDER')
            
            if folder in [upload_folder, cna_folder]:
                file_path = os.path.join(folder, filename)
                if os.path.exists(file_path):
                    thumbnail_path = generate_thumbnail(file_path)
                    if thumbnail_path:
                        image_dict['thumbnail_url'] = f"http://{server_ip}/thumbnails/{os.path.basename(thumbnail_path)}"
    
    # 如果没有生成缩略图，使用原图URL
    if 'thumbnail_url' not in image_dict and 'url' in image_dict:
        image_dict['thumbnail_url'] = image_dict['url']
    
    return image_dict 