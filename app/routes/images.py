from datetime import datetime
import PIL
from flask import Blueprint, request, jsonify,send_from_directory,current_app
from werkzeug.utils import secure_filename
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity,jwt_required
from PIL import Image as PILImage
import numpy as np
import uuid
import os
import pickle
import time
import torch
import hashlib
from ..models.models import db, User
from ..models.models import Image, Tag, ImageTag, Caption, Favorite, Download, SearchHistory, ViewHistory, News
# 添加cn_clip导入
from cn_clip.clip import tokenize
from . import images_bp
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
    index_image_vector,
    bulk_index_vectors,
    es_client
)

# 使用配置参数
Server_IP = current_app.config.get('SERVER_IP', 'localhost:5000')
# 配置参数 - 在原有Flask配置之后添加
# 修改模型和图片文件夹设置
MODEL_NAME = current_app.config.get('CLIP_MODEL_NAME', 'ViT-B-16')
CNA_FOLDER = current_app.config['CNA_FOLDER']
CACHE_DIR = current_app.config['CACHE_DIR']
IMAGE_EMBED_CACHE = current_app.config['IMAGE_EMBED_CACHE']

# 添加缩略图目录配置
THUMBNAIL_FOLDER = current_app.config['THUMBNAIL_FOLDER']
THUMBNAIL_SIZE = (200, 200)  # 缩略图尺寸

# 确保目录存在
if not os.path.exists(CNA_FOLDER):
    os.makedirs(CNA_FOLDER)
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
if not os.path.exists(THUMBNAIL_FOLDER):
    os.makedirs(THUMBNAIL_FOLDER)

# 配置路径
FLICKR_FOLDER = current_app.config['FLICKR_FOLDER']
CAPTIONS_FILE = current_app.config['CAPTIONS_FILE']
UPLOAD_FOLDER = current_app.config['UPLOAD_FOLDER']







# 添加一个通用的 OPTIONS 请求处理器
@images_bp.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response

#获取图片列表
@images_bp.route('/api/images', methods=['GET'])
def get_images():
    """获取图片列表"""
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    
    
    db_images_count = Image.query.count()
    
    
    if db_images_count == 0:
        
        image_files = [f for f in os.listdir(CNA_FOLDER) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
       
        total = len(image_files)
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total)
        current_files = image_files[start_idx:end_idx]
        
        
        images = []
        # 使用utils.py中的get_embeddings函数获取缓存的数据
        _, _, image_captions = get_embeddings()
        for filename in current_files:
            image_id = filename.split('.')[0]
            captions_list = image_captions.get(filename, ["No caption available"])
            
            image_info = {
                'id': image_id,
                'url': f"http://{Server_IP}/api/images/file/{filename}",
                'original_url': f"http://{Server_IP}/api/images/file/{filename}",
                'title': filename,
                'captions': captions_list,
                'tags': [],
                'file_path': os.path.join(CNA_FOLDER, filename)  # 添加文件路径以便生成缩略图
            }
            
            # 添加缩略图URL
            image_info = add_thumbnail_url(image_info, Server_IP)
            images.append(image_info)
        
        return jsonify({
            'total': total,
            'images': images
        })
    
    # 否则使用数据库方法
    else:
        # 从数据库获取图片
        offset = (page - 1) * per_page
        images_query = Image.query.order_by(Image.upload_date.desc()).offset(offset).limit(per_page).all()
        total_images = Image.query.count()
        
        # 构建响应
        images = []
        for img in images_query:
            captions_list = [c.caption for c in img.captions]
            tags_list = [t.name for t in img.tags]
            
            image_info = {
                'id': img.id,
                'url': img.thumbnail_url,
                'original_url': img.thumbnail_url,
                'title': img.title,
                'captions': captions_list,
                'tags': tags_list,
                'file_path': img.file_path  # 添加文件路径以便生成缩略图
            }
            
            images.append(image_info)
        
        return jsonify({
            'total': total_images,
            'images': images
        })

#提供CNA_images文件夹中的图片文件
@images_bp.route('/CNA_images/<filename>')
def get_cna_image(filename):
    """提供CNA_images文件夹中的图片文件"""
    return send_from_directory(CNA_FOLDER, filename)

#提供上传的图片文件
@images_bp.route('/uploads/<filename>')
def get_upload_image(filename):
    """提供uploads文件夹中的图片文件"""
    return send_from_directory(UPLOAD_FOLDER, filename)

#提供缩略图文件
@images_bp.route('/thumbnails/<filename>')
def get_thumbnail(filename):
    """提供缩略图文件"""
    return send_from_directory(THUMBNAIL_FOLDER, filename)


#获取单个图片详情
@images_bp.route('/api/images/<image_id>', methods=['GET'])
def get_image_detail(image_id):
    """获取单个图片详情"""
    # 检查数据库中是否有图片

    if image_id.endswith('.jpg') or image_id.endswith('.jpeg') or image_id.endswith('.png'):
        image_id = image_id.split('.')[0]

    # 记录用户浏览历史（如果用户已登录）
    try:
        # 尝试验证JWT令牌
        verify_jwt_in_request(optional=True)
        user_id = get_jwt_identity()

        # 如果用户已登录，记录浏览历史
        if user_id:
            # 检查图片是否存在
            image_exists = Image.query.get(image_id) is not None

            if image_exists:
                # 检查是否已有该图片的浏览记录，如果有则删除
                existing_view = ViewHistory.query.filter_by(user_id=user_id, image_id=image_id).first()
                if existing_view:
                    db.session.delete(existing_view)

                # 添加新的浏览记录
                new_view = ViewHistory(
                    user_id=user_id,
                    image_id=image_id
                )
                db.session.add(new_view)
                db.session.commit()
    except Exception as e:
        current_app.logger.error(f"记录浏览历史时出错: {str(e)}")
        # 错误不影响图片详情获取功能，继续执行

    db_image = Image.query.get(image_id)

    if db_image:
        # 从数据库获取图片
        captions_list = [c.caption for c in db_image.captions]

        # 获取相似图片
        similar_images = []
        if db_image.embedding:
            embedding = pickle.loads(db_image.embedding)
            similar_images = find_similar_images_by_embedding(embedding, exclude_id=image_id)

            # 为相似图片添加缩略图
            for img in similar_images:
                img = add_thumbnail_url(img, Server_IP)

        # 创建图片信息并添加缩略图URL
        image_info = {
            'id': db_image.id,
            'url': db_image.url,
            'original_url': db_image.original_url,
            'title': db_image.title,
            'captions': captions_list,
            'tags': [],
            'similar_images': similar_images,
            'file_path': db_image.file_path
        }

        image_info = add_thumbnail_url(image_info, Server_IP)

        return jsonify(image_info)

    # 如果数据库中没有，尝试从文件系统获取
    # 查找匹配的文件
    # 使用utils.py中的get_embeddings函数获取缓存的数据
    image_files_list, image_embeddings_array, image_captions = get_embeddings()
    for idx, img_file in enumerate(image_files_list):
        if img_file.split('.')[0] == image_id:
            captions_list = image_captions.get(img_file, ["No caption available"])
            tags = []

            # 获取相似图片
            similar_images = []
            if idx < len(image_embeddings_array):
                embedding = image_embeddings_array[idx]
                similar_images = find_similar_images_by_embedding(embedding, exclude_id=image_id)

                # 为相似图片添加缩略图
                for img in similar_images:
                    img = add_thumbnail_url(img, Server_IP)

            # 创建图片信息并添加缩略图URL
            image_info = {
                'id': image_id,
                'url': f"http://{Server_IP}/api/images/file/{img_file}",
                'original_url': f"http://{Server_IP}/api/images/file/{img_file}",
                'title': img_file,
                'captions': captions_list,
                'tags': tags,
                'similar_images': similar_images,
                'file_path': os.path.join(CNA_FOLDER, img_file)
            }

            image_info = add_thumbnail_url(image_info, Server_IP)

            return jsonify(image_info)

    # 如果找不到图片
    return jsonify({'error': 'Image not found'}), 404


#获取所有标签
@images_bp.route('/api/tags', methods=['GET'])
def get_tags():
    """获取所有标签"""
    # 使用utils.py中的get_embeddings函数获取缓存的数据
    _, _, image_captions = get_embeddings()
    all_tags = set()
    for captions_list in image_captions.values():
        for caption in captions_list:
            tags = []
            all_tags.update(tags)

    return jsonify(list(all_tags))


#上传图片
@images_bp.route('/api/upload', methods=['POST', 'OPTIONS'])
@jwt_required()  # 添加JWT验证
def upload_file():
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    # 检查是否有文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 检查文件类型 - 使用工具函数
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    # 获取其他参数
    title_prefix = request.form.get('title', '')
    auto_caption = request.form.get('description', '1') == '1'
    description = ""
    if auto_caption == False:
        description = request.form.get('description')

    # 获取当前用户ID
    current_user_id = get_jwt_identity()

    # 安全处理文件名
    try:
        original_filename = file.filename
        # 获取文件扩展名
        if '.' in original_filename:
            file_ext = original_filename.rsplit('.', 1)[1].lower()
        else:
            file_ext = 'jpg'  # 默认扩展名

        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        new_filename = f"{file_id}.{file_ext}"
    except Exception as e:
        current_app.logger.error(f"处理文件名时出错: {e}")
        # 出错时使用一个安全的默认值
        file_id = str(uuid.uuid4())
        new_filename = f"{file_id}.jpg"

    # 保存文件
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], new_filename)
    file.save(file_path)

    # 生成标题
    if title_prefix:
        title = f"{title_prefix}"
    else:
        title = original_filename

    # 生成图片URL
    file_url = f"http://{Server_IP}/uploads/{new_filename}"
    
    # 生成缩略图并获取缩略图URL
    thumbnail_path = generate_thumbnail(file_path, THUMBNAIL_FOLDER, THUMBNAIL_SIZE)
    thumbnail_url = None
    if thumbnail_path:
        thumbnail_url = f"http://{Server_IP}/thumbnails/{os.path.basename(thumbnail_path)}"
    else:
        # 如果缩略图生成失败，使用原图URL
        thumbnail_url = file_url

    # 计算图片嵌入向量 - 使用工具函数获取模型
    try:
        # 获取CLIP模型
        cn_clip_model, cn_preprocess = load_cn_clip_model()
        
        # 确定设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        image = cn_preprocess(PIL.Image.open(file_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = cn_clip_model.encode_image(image).cpu().numpy()[0]
        embedding_bytes = pickle.dumps(embedding)
    except Exception as e:
        current_app.logger.error(f"计算嵌入向量失败: {e}")
        embedding_bytes = None
        embedding = None

    # 创建图片记录
    new_image = Image(
        id=file_id,
        title=title,
        file_path=file_path,
        url=file_url,
        original_url=file_url,
        embedding=embedding_bytes,
        user_id=current_user_id,  # 设置用户ID
        thumbnail_url=thumbnail_url  # 设置缩略图URL
    )
    db.session.add(new_image)

    # 添加描述
    captions = []
    if auto_caption:
        caption_texts = ["这是一张自动生成描述的图片", "上传于" + datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        for caption_text in caption_texts:
            caption = Caption(caption=caption_text)
            new_image.captions.append(caption)
            captions.append(caption_text)
    else:
        caption = Caption(caption=description)
        new_image.captions.append(caption)
        captions.append(description)

    # 提交到数据库
    db.session.commit()

    # 更新内存中的嵌入向量和文件列表
    # 使用utils.py中的update_embedding函数更新缓存
    if embedding_bytes:
        embedding = pickle.loads(embedding_bytes)
        update_embedding(file_id, embedding)
        
        # 添加到 Elasticsearch
        try:
            metadata = {
                'image_id': file_id,
                'url': file_url,
                'title': title,
                'file_path': file_path,
                'captions': captions,
                'thumbnail_url': thumbnail_url  # 添加缩略图URL到元数据
            }
            index_image_vector(file_id, embedding, metadata)
            current_app.logger.info(f"图片 {file_id} 已成功索引到 Elasticsearch")
        except Exception as e:
            current_app.logger.error(f"索引图片到 Elasticsearch 失败: {str(e)}")

    return jsonify({
        'id': file_id,
        'title': title,
        'url': file_url,
        'original_url': file_url,
        'thumbnail_url': thumbnail_url,  # 在响应中包含缩略图URL
        'tags': [],
        'captions': captions,
        'message': 'File uploaded successfully'
    }), 201

#删除图片
@images_bp.route('/api/images/<image_id>', methods=['DELETE', 'OPTIONS'])
@jwt_required()
def delete_image(image_id):
    """删除图片，根据用户角色权限控制"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Methods', 'DELETE')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        return response

    # 验证用户身份
    current_user_id = get_jwt_identity()
    current_user = User.query.get(current_user_id)

    if not current_user:
        return jsonify({'error': '无效的用户身份'}), 401

    # 查找要删除的图片
    image = Image.query.get(image_id)
    if not image:
        return jsonify({'error': '图片不存在'}), 404

    # 权限检查：管理员可以删除任何图片，普通用户只能删除自己上传的图片
    if not current_user.is_admin and (image.user_id is None or image.user_id != current_user_id):
        return jsonify({'error': '权限不足，您只能删除自己上传的图片'}), 403

    try:
        # 首先删除与该图片相关的浏览记录
        ViewHistory.query.filter_by(image_id=image_id).delete()

        # 从 Elasticsearch 中删除图片向量
        try:
            if es_client:
                es_client.delete(index="image_vectors", id=image_id, ignore=[404])
                current_app.logger.info(f"已从 Elasticsearch 中删除图片 {image_id}")
        except Exception as e:
            current_app.logger.error(f"从 Elasticsearch 删除图片 {image_id} 时出错: {str(e)}")

        # 删除物理文件（如果存在且在uploads文件夹中）
        file_path = image.file_path
        if file_path and os.path.exists(file_path) and UPLOAD_FOLDER in file_path:
            try:
                os.remove(file_path)
            except Exception as e:
                current_app.logger.error(f"删除文件失败: {e}")

        # 从数据库中删除图片记录
        db.session.delete(image)
        db.session.commit()

        # 更新内存中的向量和文件列表
        load_embeddings_from_db()

        return jsonify({'success': True, 'message': '图片删除成功'})

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"删除图片时出错: {e}")
        return jsonify({'error': f'删除图片失败: {str(e)}'}), 500


# 添加批量删除图片接口
@images_bp.route('/api/images/batch-delete', methods=['POST', 'OPTIONS'])
@jwt_required()
def batch_delete_images():
    """批量删除多张图片"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        return response

    # 验证用户身份
    current_user_id = get_jwt_identity()
    current_user = User.query.get(current_user_id)

    if not current_user:
        return jsonify({'error': '无效的用户身份'}), 401

    # 获取请求数据
    data = request.get_json()
    if not data or not data.get('image_ids') or not isinstance(data.get('image_ids'), list):
        return jsonify({'error': '请提供要删除的图片ID列表'}), 400

    image_ids = data.get('image_ids')

    # 统计结果
    result = {
        'success': 0,
        'failed': 0,
        'errors': []
    }

    try:
        for image_id in image_ids:
            # 查找要删除的图片
            image = Image.query.get(image_id)
            if not image:
                result['failed'] += 1
                result['errors'].append(f"图片 {image_id} 不存在")
                continue

            # 权限检查：管理员可以删除任何图片，普通用户只能删除自己上传的图片
            if not current_user.is_admin and (image.user_id is None or image.user_id != current_user_id):
                result['failed'] += 1
                result['errors'].append(f"权限不足，无法删除图片 {image_id}")
                continue

            try:
                # 删除与该图片相关的浏览记录
                ViewHistory.query.filter_by(image_id=image_id).delete()

                # 从 Elasticsearch 中删除图片向量
                try:
                    if es_client:
                        es_client.delete(index="image_vectors", id=image_id, ignore=[404])
                        current_app.logger.info(f"已从 Elasticsearch 中删除图片 {image_id}")
                except Exception as e:
                    current_app.logger.error(f"从 Elasticsearch 删除图片 {image_id} 时出错: {str(e)}")

                # 删除物理文件（如果存在且在uploads文件夹中）
                file_path = image.file_path
                if file_path and os.path.exists(file_path) and UPLOAD_FOLDER in file_path:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        current_app.logger.error(f"删除文件失败: {e}")

                # 从数据库中删除图片记录
                db.session.delete(image)
                result['success'] += 1
            except Exception as e:
                result['failed'] += 1
                result['errors'].append(f"删除图片 {image_id} 时出错: {str(e)}")
                continue

        # 提交所有更改
        db.session.commit()

        # 更新内存中的向量和文件列表
        load_embeddings_from_db()

        return jsonify({
            'success': True,
            'message': f'成功删除 {result["success"]} 张图片，失败 {result["failed"]} 张',
            'details': result
        })

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"批量删除图片时出错: {e}")
        return jsonify({'error': f'批量删除图片失败: {str(e)}'}), 500

@images_bp.route('/api/batch-upload', methods=['POST', 'OPTIONS'])
@jwt_required()  # 添加JWT验证
def batch_upload_files():
    """批量上传图片接口，支持同时上传多张图片
    前端可以发送多个文件，标题默认为文件名，描述为空
    """
    # 处理OPTIONS请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    # 检查是否有文件
    if 'files[]' not in request.files:
        return jsonify({'error': '没有接收到文件'}), 400
    
    # 获取当前用户ID
    current_user_id = get_jwt_identity()
    
    # 获取所有文件
    files = request.files.getlist('files[]')
    
    # 验证是否有文件
    if len(files) == 0:
        return jsonify({'error': '没有选择文件'}), 400
    
    # 处理结果
    results = {
        'success': 0,
        'failed': 0,
        'total': len(files),
        'images': []
    }
    
    # 用于批量索引到 Elasticsearch 的数据
    es_vectors_data = []
    
    # 批量处理每个文件
    for file in files:
        # 检查文件名是否为空
        if file.filename == '':
            results['failed'] += 1
            continue
        
        # 检查文件类型
        if not allowed_file(file.filename):
            results['failed'] += 1
            continue
        
        try:
            # 安全处理文件名
            original_filename = file.filename
            
            # 获取文件扩展名
            if '.' in original_filename:
                file_ext = original_filename.rsplit('.', 1)[1].lower()
            else:
                file_ext = 'jpg'  # 默认扩展名
                
            # 生成唯一文件名
            file_id = str(uuid.uuid4())
            new_filename = f"{file_id}.{file_ext}"
            
            # 从原始文件名提取标题（移除路径和扩展名）
            title = os.path.basename(original_filename)
            if '.' in title:
                title = title.rsplit('.', 1)[0]
            
            # 保存文件
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], new_filename)
            file.save(file_path)
            
            # 生成图片URL
            file_url = f"http://{Server_IP}/uploads/{new_filename}"
            
            # 生成缩略图并获取缩略图URL
            thumbnail_path = generate_thumbnail(file_path, THUMBNAIL_FOLDER, THUMBNAIL_SIZE)
            thumbnail_url = None
            if thumbnail_path:
                thumbnail_url = f"http://{Server_IP}/thumbnails/{os.path.basename(thumbnail_path)}"
            else:
                # 如果缩略图生成失败，使用原图URL
                thumbnail_url = file_url
            
            # 计算图片嵌入向量
            embedding = None
            embedding_bytes = None
            try:
                cn_clip_model, cn_preprocess = load_cn_clip_model()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                image = cn_preprocess(PIL.Image.open(file_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = cn_clip_model.encode_image(image).cpu().numpy()[0]
                embedding_bytes = pickle.dumps(embedding)
            except Exception as e:
                print(f"计算嵌入向量失败: {e}")
            
            # 创建图片记录
            new_image = Image(
                id=file_id,
                title=title,
                file_path=file_path,
                url=file_url,
                original_url=file_url,
                embedding=embedding_bytes,
                user_id=current_user_id,  # 设置用户ID
                thumbnail_url=thumbnail_url  # 设置缩略图URL
            )
            db.session.add(new_image)
            
            # 添加空描述
            caption = Caption(caption="")
            new_image.captions.append(caption)
            
            # 准备 Elasticsearch 索引数据
            if embedding is not None:
                es_vectors_data.append({
                    'image_id': file_id,
                    'vector': embedding,
                    'url': file_url,
                    'title': title,
                    'file_path': file_path,
                    'captions': [""],
                    'thumbnail_url': thumbnail_url  # 添加缩略图URL到元数据
                })
            
            # 记录成功
            results['success'] += 1
            results['images'].append({
                'id': file_id,
                'title': title,
                'url': file_url,
                'original_url': file_url,
                'thumbnail_url': thumbnail_url  # 在响应中包含缩略图URL
            })
            
        except Exception as e:
            print(f"处理文件时出错: {e}")
            results['failed'] += 1
            continue
    
    # 提交到数据库（一次性提交所有成功的记录）
    if results['success'] > 0:
        db.session.commit()
        # 更新内存中的嵌入向量和文件列表
        load_embeddings_from_db()
        
        # 批量索引到 Elasticsearch
        if es_vectors_data:
            try:
                bulk_index_vectors(es_vectors_data)
                current_app.logger.info(f"成功将 {len(es_vectors_data)} 张图片索引到 Elasticsearch")
            except Exception as e:
                current_app.logger.error(f"批量索引到 Elasticsearch 失败: {str(e)}")
    
    # 返回处理结果
    return jsonify({
        'success': True,
        'message': f'批量上传完成: 成功 {results["success"]} 张，失败 {results["failed"]} 张',
        'results': results
    }), 201



#更新图片
@images_bp.route('/api/images/<image_id>', methods=['PUT', 'OPTIONS'])
@jwt_required()
def update_image(image_id):
    """更新图片信息，包括标题和描述"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Methods', 'PUT')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        return response

    # 验证用户身份
    current_user_id = get_jwt_identity()
    current_user = User.query.get(current_user_id)

    if not current_user:
        return jsonify({'error': '无效的用户身份'}), 401

    # 查找要更新的图片
    image = Image.query.get(image_id)
    if not image:
        return jsonify({'error': '图片不存在'}), 404

    # 权限检查：管理员可以更新任何图片，普通用户只能更新自己上传的图片
    if not current_user.is_admin and (image.user_id is None or image.user_id != current_user_id):
        return jsonify({'error': '权限不足，您只能编辑自己上传的图片'}), 403

    # 获取请求数据
    data = request.get_json()

    try:
        # 更新标题
        if 'title' in data:
            image.title = data['title']

        # 更新描述
        captions_list = []
        if 'captions' in data:
            # 删除现有描述
            Caption.query.filter_by(image_id=image_id).delete()

            # 添加新描述
            for caption_text in data['captions']:
                if caption_text.strip():  # 忽略空字符串
                    caption = Caption(image_id=image_id, caption=caption_text)
                    db.session.add(caption)
                    captions_list.append(caption_text)

        # 提交更改
        db.session.commit()

        # 更新内存中的嵌入向量和文件列表
        load_embeddings_from_db()
        
        # 更新 Elasticsearch 中的数据
        try:
            # 检查图片是否有嵌入向量
            if image.embedding:
                embedding = pickle.loads(image.embedding)
                
                # 获取更新后的元数据
                metadata = {
                    'image_id': image.id,
                    'url': image.url,
                    'title': image.title,
                    'file_path': image.file_path,
                    'captions': captions_list if captions_list else [c.caption for c in image.captions]
                }
                
                # 更新 Elasticsearch 索引
                index_image_vector(image.id, embedding, metadata)
                current_app.logger.info(f"已更新 Elasticsearch 中图片 {image.id} 的信息")
        except Exception as e:
            current_app.logger.error(f"更新 Elasticsearch 中图片信息失败: {str(e)}")

        # 获取更新后的图片信息
        captions_list = [c.caption for c in image.captions]
        tags_list = [t.name for t in image.tags]

        return jsonify({
            'message': '图片信息更新成功',
            'image': {
                'id': image.id,
                'url': image.url,
                'original_url': image.original_url,
                'title': image.title,
                'captions': captions_list,
                'tags': tags_list
            }
        })

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"更新图片时出错: {e}")
        return jsonify({'error': f'更新图片失败: {str(e)}'}), 500


