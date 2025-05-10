from flask import Flask, request, jsonify, send_from_directory
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, verify_jwt_in_request
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import os
import json
import torch
import numpy as np
import PIL
from PIL import Image
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import time
from transformers import AutoTokenizer, AutoModel, logging as transformers_logging
import warnings
import uuid
from datetime import datetime
import logging



#导入模型
import clip
from models import db, User, Image, Tag, ImageTag, Caption, Favorite, Download, SearchHistory, ViewHistory, News

#导入新闻推荐相关函数
from novo2 import recommend_articles, get_article_weights, set_article_weights
from novo2 import get_time_range, set_time_range
from novo1 import ImageRecommender,NewsAnalyzer, get_current_weights, set_weights

# 添加cn_clip导入
from cn_clip.clip import load_from_name, tokenize
import uuid
import pickle
from PIL import Image as PILImage
import time


device = "cuda" if torch.cuda.is_available() else "cpu"

# 配置参数 - 在原有Flask配置之后添加
# 修改模型和图片文件夹设置
MODEL_NAME = "ViT-B-16"
CNA_FOLDER = 'CNA_images'
CACHE_DIR = "CNA_cache"
IMAGE_EMBED_CACHE = os.path.join(CACHE_DIR, "CNA_image_embeddings.npz")

# 确保目录存在
if not os.path.exists(CNA_FOLDER):
    os.makedirs(CNA_FOLDER)
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# 初始化cn_clip模型
cn_clip_model, cn_preprocess = None, None
try:
    cn_clip_model, cn_preprocess = load_from_name(MODEL_NAME, device=device)
    cn_clip_model.eval()
    print(f"CN-CLIP模型({MODEL_NAME})加载成功")
except Exception as e:
    print(f"CN-CLIP模型加载失败: {e}")

# 抑制各种警告
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()  # 只显示错误，不显示警告
logging.getLogger("PIL").setLevel(logging.ERROR)  # 抑制PIL警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制TensorFlow警告

# 配置Flask日志级别
app = Flask(__name__)
app.logger.setLevel(logging.ERROR)  # 只显示错误日志
CORS(app)

#服务器IP地址
Server_IP="localhost:5000"



# 配置JWT
app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # 在生产环境中使用安全的密钥
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 30 * 24 * 60 * 60  # 30天过期时间
jwt = JWTManager(app)


#配置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")


# 配置路径
FLICKR_FOLDER = 'Flickr8K/Images'
CAPTIONS_FILE = 'Flickr8K/captions.txt'
UPLOAD_FOLDER = 'uploads'






# 确保目录存在
for folder in [ UPLOAD_FOLDER, CNA_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 配置数据库连接
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost/pixtock?charset=utf8mb4&init_command=SET%20time_zone%3D%27%2B08%3A00%27'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
print("数据库连接成功")


#--------------------------------------------------------------------------#
# 加载图片嵌入和文件列表
image_files_list = []
image_embeddings_array = np.zeros((0, 512), dtype=np.float32)
image_captions = {}  # 改为从数据库加载的图片描述

#加载新闻分析和图片推荐器
news_analyzer = NewsAnalyzer()
image_recommender = ImageRecommender()

def load_embeddings_from_db():
    """从数据库加载图片嵌入向量和描述"""
    global image_files_list, image_embeddings_array, image_captions
    try:
        # 从数据库获取所有图片及其嵌入向量
        all_images = Image.query.all()
        image_files_list = []
        embeddings_list = []
        image_captions = {}  # 重置图片描述字典
        
        for img in all_images:
            if img.embedding:
                try:
                    embedding = pickle.loads(img.embedding)
                    image_files_list.append(img.id + '.jpg')  # 假设所有图片都是jpg格式
                    embeddings_list.append(embedding)
                except Exception as e:
                    print(f"处理图片 {img.id} 的嵌入向量时出错: {e}")
                    continue
        
        # 转换为numpy数组
        image_embeddings_array = np.array(embeddings_list)
        print(f"已从数据库加载 {len(image_files_list)} 张图片的嵌入向量")
    except Exception as e:
        print(f"从数据库加载图片嵌入失败: {e}")
        # 初始化为空数组，以防加载失败
        image_files_list = []
        image_embeddings_array = np.zeros((0, 512), dtype=np.float32)
        image_captions = {}

# 添加新函数：初始化图片数据库
def init_image_db_from_cna():
    """从CNA_images文件夹初始化图片数据库表"""
    if not os.path.exists(CNA_FOLDER):
        print(f"错误: 找不到CNA图片文件夹 ({CNA_FOLDER})")
        return False
    
    if cn_clip_model is None:
        print("错误: CN-CLIP模型未加载")
        return False
    
    print("开始初始化图片数据库...")
    start_time = time.time()
    
    # 清空相关表
    try:

        print("数据库表已清空")
    except Exception as e:
        print(f"清空表时出错: {e}")
        db.session.rollback()
        return False
    
    # 获取图片文件列表
    image_files = [f for f in os.listdir(CNA_FOLDER) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    # 预先计算图片嵌入向量
    if os.path.exists(IMAGE_EMBED_CACHE):
        # 从缓存加载预计算的向量
        print("从缓存加载图片嵌入向量...")
        with np.load(IMAGE_EMBED_CACHE) as data:
            cached_files = data['filenames'].tolist()
            cached_embeddings = data['embeddings']
            
            # 创建文件名到嵌入向量的映射
            embeddings_dict = {f: emb for f, emb in zip(cached_files, cached_embeddings)}
    else:
        # 如果没有缓存，则需要即时计算
        embeddings_dict = {}
    
    # 插入图片记录
    for i, filename in enumerate(image_files):
        try:
            # 生成UUID作为图片ID
            image_id = str(uuid.uuid4())
            
            # 构造URL
            file_url = f"http://{Server_IP}/CNA_images/{filename}"
            
            # 获取或计算嵌入向量
            if filename in embeddings_dict:
                embedding = embeddings_dict[filename]
                embedding_bytes = pickle.dumps(embedding)
            else:
                # 加载并转换图片
                image_path = os.path.join(CNA_FOLDER, filename)
                image = PILImage.open(image_path).convert("RGB")
                
                # 计算嵌入向量
                with torch.no_grad():
                    inputs = cn_preprocess(image).unsqueeze(0).to(device)
                    embedding = cn_clip_model.encode_image(inputs).cpu().numpy()[0]
                
                embedding_bytes = pickle.dumps(embedding)
            
            # 创建图片记录
            new_image = Image(
                id=image_id,
                title=filename,
                file_path=os.path.join(CNA_FOLDER, filename),
                url=file_url,
                original_url=file_url,
                embedding=embedding_bytes
            )
            
            db.session.add(new_image)
            
            # 每100条记录提交一次，避免事务过大
            if (i + 1) % 100 == 0 or i == len(image_files) - 1:
                db.session.commit()
                print(f"已处理 {i+1}/{len(image_files)} 张图片")
                
        except Exception as e:
            print(f"处理图片 {filename} 时出错: {e}")
            continue
    
    db.session.commit()
    elapsed_time = time.time() - start_time
    print(f"数据库初始化完成，共处理 {len(image_files)} 张图片，耗时 {elapsed_time:.2f} 秒")
    
    # 加载嵌入向量到内存
    load_embeddings_from_db()
    return True


def find_similar_images_by_embedding(embedding, exclude_id=None, max_results=20):
    """根据嵌入向量查找相似图片（数据库版本）"""
    similar_images = []
    
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
                print(f"处理图片 {img.id} 的嵌入向量时出错: {e}")
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
    for idx in top_indices:
        img = images[idx]
        score = float(similarities[idx])
        captions_list = []
        
        similar_images.append({
            'id': img.id,
            'url': img.url,
            'original_url': img.original_url,
            'title': img.title,
            'captions': captions_list,
            'tags': [],
            'score': score
        })
    
    return similar_images




@app.route('/api/images', methods=['GET'])
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
        for filename in current_files:
            image_id = filename.split('.')[0]
            captions_list = image_captions.get(filename, ["No caption available"])
            
            images.append({
                'id': image_id,
                'url': f"http://{Server_IP}/api/images/file/{filename}",
                'original_url': f"http://{Server_IP}/api/images/file/{filename}",
                'title': filename,
                'captions': captions_list,
                'tags': []
            })
        
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
            
            images.append({
                'id': img.id,
                'url': img.url,
                'original_url': img.original_url,
                'title': img.title,
                'captions': captions_list,
                'tags': tags_list
            })
        
        return jsonify({
            'total': total_images,
            'images': images
        })


@app.route('/CNA_images/<filename>')
def get_cna_image(filename):
    """提供CNA_images文件夹中的图片文件"""
    return send_from_directory(CNA_FOLDER, filename)

@app.route('/uploads/<filename>')
def get_upload_image(filename):
    """提供uploads文件夹中的图片文件"""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/api/images/search', methods=['GET'])
def search_images():
    """通过关键词搜索图片 (结合文本匹配和CN-CLIP模型查找相似图片)"""
    start_time = time.time()
    print(f"开始搜索...")
    
    query = request.args.get('query', '')
    
    if not query:
        return jsonify({'error': '搜索关键词不能为空'}), 400
    
    # 记录用户检索历史（如果用户已登录）
    try:
        # 尝试验证JWT令牌
        verify_jwt_in_request(optional=True)
        user_id = get_jwt_identity()
        print(f"用户ID: {user_id}")
        # 如果用户已登录，记录检索历史
        if user_id:
            new_search = SearchHistory(
                user_id=user_id,
                search_content=query
            )
            db.session.add(new_search)
            db.session.commit()
    except Exception as e:
        print(f"记录检索历史时出错: {str(e)}")
        # 错误不影响搜索功能，继续执行
    
    try:
        # 第一步：从数据库查找描述中包含搜索关键词的图片
        relevant_images = []
        description_matches = db.session.query(Image).join(Caption).filter(
            Caption.caption.like(f"%{query}%")
        ).all()
        
        # 记录匹配描述的图片ID，避免重复
        matched_image_ids = set()
        for img in description_matches:
            matched_image_ids.add(img.id)
            # 添加到结果中，相似度设为1.0（最高）表示直接匹配
            relevant_images.append({
                'id': img.id,
                'url': img.url,
                'original_url': img.original_url,
                'title': img.title,
                'captions': [c.caption for c in img.captions],
                'tags': [],
                'score': 1.0,
                'match_type': '描述匹配'
            })
        
        # 第二步：使用CN-CLIP模型搜索语义相关图片
        text_inputs = tokenize([query]).to(device)
        with torch.no_grad():
            text_embedding = cn_clip_model.encode_text(text_inputs).cpu().numpy()
        
        # 计算相似度
        similarities = cosine_similarity(text_embedding, image_embeddings_array)[0]
        
        # 获取排序后的索引
        top_k = 100  # 返回前100个结果
        indices = np.argsort(-similarities)[:top_k]
        
        # 构建余下搜索结果（不包括已经通过描述匹配的图片）
        for idx in indices:
            img_file = image_files_list[idx]
            score = float(similarities[idx])
            
            # 获取图片ID
            image_id = img_file.split('.')[0]
            
            # 跳过已经通过描述匹配的图片
            if image_id in matched_image_ids:
                continue
            
            # 从数据库获取图片信息
            image = Image.query.get(image_id)
            
            if image:
                relevant_images.append({
                    'id': image_id,
                    'url': image.url,
                    'original_url': image.original_url,
                    'title': image.title,
                    'captions': [c.caption for c in image.captions],
                    'tags': [],
                    'score': score,
                    'match_type': '语义相关'
                })
        
        # 按相似度降序排序结果
        search_results = sorted(relevant_images, key=lambda x: x['score'], reverse=True)
        
        # 移除match_type字段，保持API兼容性
        for item in search_results:
            if 'match_type' in item:
                del item['match_type']
        
        print(f"搜索完成，找到 {len(search_results)} 个结果，耗时: {time.time() - start_time:.4f}秒")
        
        # 返回搜索结果
        return jsonify({
            'total': len(search_results),
            'images': search_results
        })
        
    except Exception as e:
        print(f"图片搜索出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'搜索处理失败: {str(e)}'}), 500


@app.route('/api/images/<image_id>/similar', methods=['GET'])
def get_similar_images(image_id):
    """获取与指定图片相似的图片列表"""
    # 开始计时
    total_start_time = time.perf_counter()
    print(f"开始查找与图片 {image_id} 相似的图片...")
    
    # 清理ID，去掉可能的图片扩展名
    if image_id.endswith('.jpg') or image_id.endswith('.png'):
        image_id = image_id.split('.')[0]
    
    # 获取目标图片的嵌入
    target_image = Image.query.get(image_id)
    if not target_image or not target_image.embedding:
        return jsonify({'error': f'图片 {image_id} 不存在或没有嵌入向量'}), 404
    
    # 获取目标图片的嵌入向量
    target_embedding = pickle.loads(target_image.embedding)
    
    # 计算相似度
    target_embedding_normalized = target_embedding / np.linalg.norm(target_embedding)
    all_embeddings_normalized = image_embeddings_array / np.linalg.norm(image_embeddings_array, axis=1, keepdims=True)
    similarities = np.dot(all_embeddings_normalized, target_embedding_normalized)
    
    # 获取相似度最高的图片（不包括自己）
    top_k = 20
    sorted_indices = np.argsort(-similarities)[:top_k + 1]
    
    # 查找目标图片的索引位置
    target_idx = None
    for idx, img_file in enumerate(image_files_list):
        if img_file.startswith(image_id):
            target_idx = idx
            break
    
    # 如果目标图片在结果中，将其排除
    if target_idx is not None:
        sorted_indices = [idx for idx in sorted_indices if idx != target_idx][:top_k]
    
    # 构建结果
    similar_images = []
    for idx in sorted_indices:
        img_file = image_files_list[idx]
        score = float(similarities[idx])
        
        # 获取图片ID
        result_id = img_file.split('.')[0]
        
        # 从数据库获取图片信息
        image = Image.query.get(result_id)
        
        if image:
            similar_images.append({
                'id': result_id,
                'url': image.url,
                'original_url': image.original_url,
                'title': image.title,
                'captions': [],
                'tags': [],
                'score': score
            })
    
    # 统计总耗时
    total_time = time.perf_counter() - total_start_time
    print(f"相似图片搜索完成，找到 {len(similar_images)} 个结果")
    print(f"总耗时: {total_time:.4f}秒")
    
    # 返回结果
    return jsonify({
        'similar_images': similar_images,
        'total': len(similar_images)
    })

@app.route('/api/images/<image_id>', methods=['GET'])
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
        print(f"记录浏览历史时出错: {str(e)}")
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
        
        return jsonify({
            'id': db_image.id,
            'url': db_image.url,
            'original_url': db_image.original_url,
            'title': db_image.title,
            'captions': captions_list,
            'tags': [],
            'similar_images': similar_images
        })
    
    # 如果数据库中没有，尝试从文件系统获取
    # 查找匹配的文件
    for idx, img_file in enumerate(image_files_list):
        if img_file.split('.')[0] == image_id:
            captions_list = image_captions.get(img_file, ["No caption available"])
            tags = []
            
            # 获取相似图片
            similar_images = []
            if idx < len(image_embeddings_array):
                embedding = image_embeddings_array[idx]
                similar_images = find_similar_images_by_embedding(embedding, exclude_id=image_id)
            
            return jsonify({
                'id': image_id,
                'url': f"http://{Server_IP}/api/images/file/{img_file}",
                'original_url': f"http://{Server_IP}/api/images/file/{img_file}",
                'title': img_file,
                'captions': captions_list,
                'tags': tags,
                'similar_images': similar_images
            })
    
    # 如果找不到图片
    return jsonify({'error': 'Image not found'}), 404

@app.route('/api/tags', methods=['GET'])
def get_tags(): 
    """获取所有标签"""
    all_tags = set()
    for captions_list in image_captions.values():
        for caption in captions_list:
            tags = []
            all_tags.update(tags)
    
    return jsonify(list(all_tags))



@app.route('/api/upload', methods=['POST', 'OPTIONS'])
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
    
    # 检查文件类型
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
        print(f"处理文件名时出错: {e}")
        # 出错时使用一个安全的默认值
        file_id = str(uuid.uuid4())
        new_filename = f"{file_id}.jpg"
    
    # 保存文件
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    file.save(file_path)
    
    # 生成标题
    if title_prefix:
        title = f"{title_prefix}"
    else:
        title = original_filename
    
    # 生成图片URL
    file_url = f"http://{Server_IP}/uploads/{new_filename}"
    
    # 计算图片嵌入向量
    try:
        image = cn_preprocess(PIL.Image.open(file_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = cn_clip_model.encode_image(image).cpu().numpy()[0]
        embedding_bytes = pickle.dumps(embedding)
    except Exception as e:
        print(f"计算嵌入向量失败: {e}")
        embedding_bytes = None
    
    # 创建图片记录
    new_image = Image(
        id=file_id,
        title=title,
        file_path=file_path,
        url=file_url,
        original_url=file_url,
        embedding=embedding_bytes,
        user_id=current_user_id  # 设置用户ID
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
    load_embeddings_from_db()
    
    return jsonify({
        'id': file_id,
        'title': title,
        'url': file_url,
        'original_url': file_url,
        'tags': [],
        'captions': captions,
        'message': 'File uploaded successfully'
    }), 201

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# 用户注册
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    
    # 验证必要字段
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Missing username or password'}), 400
    
    # 检查用户名是否已存在
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 409
    
    # 创建新用户
    hashed_password = generate_password_hash(data['password'])
    new_user = User(
        username=data['username'],
        password=hashed_password,
        email=data.get('email')
    )
    
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({'message': 'User registered successfully'}), 201

# 用户登录
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    user = User.query.filter_by(username=username).first()
    
    if not user or not check_password_hash(user.password, password):
        return jsonify({'error': 'Invalid username or password'}), 401
    
    # 创建 JWT 令牌，确保 subject 是字符串
    access_token = create_access_token(identity=str(user.id))
    
    return jsonify({
        'access_token': access_token,
        'user': {
            'id': user.id,
            'username': user.username,
            'role': user.role,
            'email': user.email,
            'is_admin': user.is_admin
        }
    }), 200

# 获取当前用户信息
@app.route('/api/user', methods=['GET'])
@jwt_required()
def get_user():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'role': user.role,
        'is_admin': user.is_admin
    }), 200

# 修改收藏添加函数
@app.route('/api/favorites', methods=['POST', 'OPTIONS'])
def add_favorite():
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    # 验证 JWT
    try:
        print("验证 JWT")
        verify_jwt_in_request()
        user_id = int(get_jwt_identity())  # 将字符串 ID 转换为整数
    except Exception as e:
        return jsonify({'error': 'Authentication required', 'details': str(e)}), 401
    
    data = request.get_json()
    
    if not data or not data.get('image_id'):
        return jsonify({'error': 'Image ID is required'}), 400
    
    image_id = data['image_id']
    if image_id.endswith('.png') or image_id.endswith('.jpg') or image_id.endswith('.jpeg'):
        image_id = image_id.split('.')[0]
    # 检查图片是否存在
    image = Image.query.get(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404
    
    # 检查是否已经收藏
    existing_favorite = Favorite.query.filter_by(user_id=user_id, image_id=image_id).first()
    if existing_favorite:
        return jsonify({'message': 'Image already in favorites'}), 200
    
    # 添加收藏
    try:
        new_favorite = Favorite(user_id=user_id, image_id=image_id)
        db.session.add(new_favorite)
        db.session.commit()
        return jsonify({'message': 'Image added to favorites'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to add favorite: {str(e)}'}), 500

# 修改获取收藏列表的API
@app.route('/api/favorites', methods=['GET', 'OPTIONS'])
def get_favorites():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        return response
    
    try:
        # 验证 JWT
        try:
            verify_jwt_in_request()
            user_id = int(get_jwt_identity())
        except Exception as e:
            return jsonify({'error': 'Authentication required', 'details': str(e)}), 401
        
        # 获取用户的所有收藏，不使用分页
        favorites_query = Favorite.query.filter_by(user_id=user_id).join(
            Image, Favorite.image_id == Image.id
        ).order_by(Favorite.added_at.desc()).all()
        
        # 构建响应
        favorites = []
        for fav in favorites_query:
            image = Image.query.get(fav.image_id)
            if image:
                captions_list = [c.caption for c in image.captions]
                tags_list = [t.name for t in image.tags]
                
                favorites.append({
                    'id': image.id,
                    'url': image.url,
                    'original_url': image.original_url,
                    'title': image.title,
                    'captions': captions_list,
                    'tags': tags_list,
                    'added_at': fav.added_at.isoformat()
                })
        
        return jsonify({
            'favorites': favorites,
            'total': len(favorites)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 修改删除收藏函数
@app.route('/api/favorites/<image_id>', methods=['DELETE', 'OPTIONS'])
@jwt_required(optional=True)
def remove_favorite(image_id):
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'DELETE,OPTIONS')
        return response
    
    # 原有的 DELETE 请求处理逻辑
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({'error': 'Authentication required'}), 401
    
    if image_id.endswith('.png') or image_id.endswith('.jpg') or image_id.endswith('.jpeg'):
        image_id = image_id.split('.')[0]
    # 查找收藏
    favorite = Favorite.query.filter_by(user_id=user_id, image_id=image_id).first()
    if not favorite:
        return jsonify({'error': 'Favorite not found'}), 404
    
    # 删除收藏
    db.session.delete(favorite)
    db.session.commit()
    
    return jsonify({'message': 'Favorite removed successfully'}), 200

# 修改下载记录添加函数
@app.route('/api/downloads', methods=['POST', 'OPTIONS'])
@jwt_required(optional=True)  # 使 OPTIONS 请求不需要 JWT
def add_download():
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    # 原有的 POST 请求处理逻辑
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({'error': 'Authentication required'}), 401
        
    data = request.get_json()
    
    if not data or not data.get('image_id'):
        return jsonify({'error': 'Image ID is required'}), 400
    
    image_id = data['image_id']
    if image_id.endswith('.png') or image_id.endswith('.jpg') or image_id.endswith('.jpeg'):
        image_id = image_id.split('.')[0]
    
    # 检查图片是否存在
    image = Image.query.get(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404
    
    # 检查是否已有该图片的下载记录，如果有则删除
    existing_download = Download.query.filter_by(user_id=user_id, image_id=image_id).first()
    if existing_download:
        db.session.delete(existing_download)
        db.session.commit()
    
    # 添加新的下载记录
    new_download = Download(user_id=user_id, image_id=image_id)
    db.session.add(new_download)
    db.session.commit()
    
    return jsonify({'message': 'Download recorded successfully'}), 201

# 修改获取下载历史的API
@app.route('/api/downloads', methods=['GET', 'OPTIONS'])
@jwt_required(optional=True)
def get_downloads():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        return response
    
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        # 获取用户的所有下载记录，按时间倒序排序
        downloads_query = Download.query.filter_by(user_id=user_id).order_by(
            Download.downloaded_at.desc()
        ).all()
        
        # 使用集合来跟踪已处理的图片ID，确保每个图片只显示最新的下载记录
        processed_image_ids = set()
        downloads = []
        
        for dl in downloads_query:
            # 如果这个图片ID已经处理过了，跳过
            if dl.image_id in processed_image_ids:
                continue
                
            # 标记这个图片ID为已处理
            processed_image_ids.add(dl.image_id)
            
            # 获取图片信息
            image = Image.query.get(dl.image_id)
            if image:
                captions_list = [c.caption for c in image.captions]
                tags_list = [t.name for t in image.tags]
                
                downloads.append({
                    'id': image.id,
                    'url': image.url,
                    'original_url': image.original_url,
                    'title': image.title,
                    'captions': captions_list,
                    'tags': tags_list,
                    'downloaded_at': dl.downloaded_at.isoformat()
                })
        
        return jsonify({
            'downloads': downloads,
            'total': len(downloads)
        })
    except Exception as e:
        print(f"获取下载历史时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 修改清空下载历史函数
@app.route('/api/downloads', methods=['DELETE', 'OPTIONS'])
@jwt_required(optional=True)
def clear_downloads():
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'DELETE,OPTIONS')
        return response
    
    # 原有的 DELETE 请求处理逻辑
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({'error': 'Authentication required'}), 401
    
    # 删除用户的所有下载记录
    Download.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    
    return jsonify({'message': 'Download history cleared successfully'}), 200

# 添加一个通用的 OPTIONS 请求处理器
@app.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response 

# 修改 Novo 平台分析新闻接口
@app.route('/api/novo/get_recommendations_news', methods=['POST', 'OPTIONS'])
def novo_get_recommendations_news():
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
        
    # 处理 POST 请求
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "无效的请求数据"}), 400
            
        title = data.get('title', '')
        content = data.get('content', '')
        
        if not title or not content:
            return jsonify({"error": "标题和内容不能为空"}), 400
        
        # 记录开始时间
        start_time = time.perf_counter()
        
        # 调用推荐函数，使用数据库加载文章
        recommendations = recommend_articles(
            title, 
            content, 
            db_session=db.session, 
            News=News, 
            top_n=5,
            use_db=True  # 指定使用数据库
        )
        
        # 记录结束时间
        end_time = time.perf_counter()
        print(f"推荐新闻耗时: {end_time - start_time:.4f}秒")
        
        # 确保所有数值都是Python原生类型
        for rec in recommendations:
            rec['id'] = int(rec['id'])  # 将numpy.int64转换为Python int
            rec['similarity'] = float(rec['similarity'])  # 将numpy.float64转换为Python float
            
            # 根据数据库模式，可能会返回topic和sentiment而不是theme和label
            if 'theme' in rec and isinstance(rec['theme'], np.integer):
                rec['theme'] = int(rec['theme'])
            if 'label' in rec and isinstance(rec['label'], np.integer):
                rec['label'] = int(rec['label'])
        
        # 构建响应
        return jsonify(recommendations)
        
    except Exception as e:
        print(f"获取推荐新闻失败: {str(e)}")
        import traceback
        print(traceback.format_exc())  # 打印完整堆栈跟踪
        return jsonify({"error": f"获取推荐新闻失败: {str(e)}"}), 500

@app.route('/api/novo/get_recommendations_images', methods=['POST', 'OPTIONS'])
def novo_get_recommendations_images():
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
        
    # 处理 POST 请求
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "无效的请求数据"}), 400
            
        title = data.get('title', '')
        content = data.get('content', '')
        
        if not title:
            return jsonify({"error": "标题不能为空"}), 400
        
        news_elements=news_analyzer.analyze_news(title,content)
        image_recommendations=image_recommender.recommend_images(news_elements,title,content,use_fixed_weights=True)
        features = {
        "title": title,
        "事件内容": news_elements["核心事件"],
        "实体内容": news_elements["显著实体"],
        "数据内容": news_elements["数据统计"],
        "动作内容": news_elements["关键动作"],
        "场景内容": news_elements["场景特征"],
        "情感内容": news_elements["情感内容"],
        "隐喻内容": news_elements["视觉隐喻"],
        }
        recommendations={
            "news_id":"",
            "news_title":"",
            "features":features,
            "recommended_images":[{
                "image_path":img_url,
                "similarity_score":float(score)
            }
            for img_url, score in image_recommendations
            ]
        }

        
        # 构建响应
        return jsonify(recommendations)
        
    except Exception as e:
        print(f"获取推荐图片失败: {str(e)}")
        return jsonify({"error": f"获取推荐图片失败: {str(e)}"}), 500


# 在应用启动前调用初始化函数
def migrate_image_urls():
    """迁移数据库中的图片URL，将localhost:5000替换为当前Server_IP"""
    try:
        print(f"开始迁移数据库中的图片URL...")
        images = Image.query.all()
        update_count = 0
        
        for img in images:
            # 检查并更新URL
            if 'localhost:5000' in img.url:
                img.url = img.url.replace('localhost:5000', Server_IP)
                update_count += 1
                
            # 检查并更新原始URL
            if 'localhost:5000' in img.original_url:
                img.original_url = img.original_url.replace('localhost:5000', Server_IP)
                
        if update_count > 0:
            db.session.commit()
            print(f"成功更新 {update_count} 条图片记录的URL")
        else:
            print("没有发现需要更新的URL")
    except Exception as e:
        print(f"迁移图片URL时出错: {e}")
        db.session.rollback()

# 记录用户检索记录
@app.route('/api/search_history', methods=['POST', 'OPTIONS'])
@jwt_required(optional=True)
def add_search_history():
    """记录用户的检索记录"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    # 获取用户ID
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({'error': 'Authentication required'}), 401
    
    # 获取请求数据
    data = request.get_json()
    if not data or not data.get('search_content'):
        return jsonify({'error': 'Search content is required'}), 400
    
    # 记录检索历史
    search_content = data.get('search_content')
    
    # 创建新记录
    try:
        new_search = SearchHistory(
            user_id=user_id,
            search_content=search_content
        )
        db.session.add(new_search)
        db.session.commit()
        
        return jsonify({
            'message': 'Search history recorded successfully',
            'id': new_search.id
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to record search history: {str(e)}'}), 500

# 记录用户浏览记录
@app.route('/api/view_history', methods=['POST', 'OPTIONS'])
@jwt_required(optional=True)
def add_view_history():
    """记录用户的图片浏览记录"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    # 获取用户ID
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({'error': 'Authentication required'}), 401
    
    # 获取请求数据
    data = request.get_json()
    if not data or not data.get('image_id'):
        return jsonify({'error': 'Image ID is required'}), 400
    
    # 获取图片ID
    image_id = data.get('image_id')
    if image_id.endswith('.png') or image_id.endswith('.jpg') or image_id.endswith('.jpeg'):
        image_id = image_id.split('.')[0]
    
    # 检查图片是否存在
    image = Image.query.get(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404
    
    # 创建或更新浏览记录
    try:
        # 检查是否已有该图片的浏览记录，如果有则删除原来的
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
        
        return jsonify({
            'message': 'View history recorded successfully',
            'id': new_view.id
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to record view history: {str(e)}'}), 500

# 获取用户的检索历史
@app.route('/api/search_history', methods=['GET', 'OPTIONS'])
@jwt_required(optional=True)
def get_search_history():
    """获取用户的检索历史"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        return response
    
    # 获取用户ID
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({'error': 'Authentication required'}), 401
    
    # 获取分页参数
    limit = request.args.get('limit', 20, type=int)
    
    # 查询用户的检索历史
    try:
        history = SearchHistory.query.filter_by(user_id=user_id).order_by(
            SearchHistory.added_at.desc()
        ).limit(limit).all()
        
        search_history = []
        for item in history:
            search_history.append({
                'id': item.id,
                'search_content': item.search_content,
                'added_at': item.added_at.isoformat()
            })
        
        return jsonify({
            'search_history': search_history,
            'total': len(search_history)
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch search history: {str(e)}'}), 500

# 获取用户的浏览历史
@app.route('/api/view_history', methods=['GET', 'OPTIONS'])
@jwt_required(optional=True)
def get_view_history():
    """获取用户的图片浏览历史"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        return response
    
    # 获取用户ID
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({'error': 'Authentication required'}), 401
    
    # 获取分页参数
    limit = request.args.get('limit', 20, type=int)
    
    # 查询用户的浏览历史
    try:
        views = ViewHistory.query.filter_by(user_id=user_id).join(
            Image, ViewHistory.image_id == Image.id
        ).order_by(ViewHistory.added_at.desc()).limit(limit).all()
        
        view_history = []
        for view in views:
            image = Image.query.get(view.image_id)
            if image:
                captions_list = [c.caption for c in image.captions]
                tags_list = [t.name for t in image.tags]
                
                view_history.append({
                    'id': image.id,
                    'url': image.url,
                    'original_url': image.original_url,
                    'title': image.title,
                    'captions': captions_list,
                    'tags': tags_list,
                    'viewed_at': view.added_at.isoformat()
                })
        
        return jsonify({
            'view_history': view_history,
            'total': len(view_history)
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch view history: {str(e)}'}), 500

# 添加个性化推荐接口
@app.route('/api/personalized_recommendations', methods=['GET', 'OPTIONS'])
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
        from RecSys import get_recommender_instance
        
        # 获取推荐系统实例
        recommender = get_recommender_instance(app)
        
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

@app.route('/api/images/<image_id>', methods=['DELETE', 'OPTIONS'])
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
        
        # 删除图片关联的记录
        # 因为设置了级联删除，相关联的标签和描述会自动删除
        
        # 删除物理文件（如果存在且在uploads文件夹中）
        file_path = image.file_path
        if file_path and os.path.exists(file_path) and UPLOAD_FOLDER in file_path:
            try:
                os.remove(file_path)
            except Exception as e:
                app.logger.error(f"删除文件失败: {e}")
        
        # 从数据库中删除图片记录
        db.session.delete(image)
        db.session.commit()
        
        # 更新内存中的向量和文件列表
        load_embeddings_from_db()
        
        return jsonify({'success': True, 'message': '图片删除成功'})
    
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"删除图片时出错: {e}")
        return jsonify({'error': f'删除图片失败: {str(e)}'}), 500

# 添加批量删除图片接口
@app.route('/api/images/batch-delete', methods=['POST', 'OPTIONS'])
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
                
                # 删除物理文件（如果存在且在uploads文件夹中）
                file_path = image.file_path
                if file_path and os.path.exists(file_path) and UPLOAD_FOLDER in file_path:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        app.logger.error(f"删除文件失败: {e}")
                
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
        app.logger.error(f"批量删除图片时出错: {e}")
        return jsonify({'error': f'批量删除图片失败: {str(e)}'}), 500


@app.route('/api/images/<image_id>', methods=['PUT', 'OPTIONS'])
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
        if 'captions' in data:
            # 删除现有描述
            Caption.query.filter_by(image_id=image_id).delete()
            
            # 添加新描述
            for caption_text in data['captions']:
                if caption_text.strip():  # 忽略空字符串
                    caption = Caption(image_id=image_id, caption=caption_text)
                    db.session.add(caption)
        
        # 提交更改
        db.session.commit()
        
        # 更新内存中的嵌入向量和文件列表
        load_embeddings_from_db()
        
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
        app.logger.error(f"更新图片时出错: {e}")
        return jsonify({'error': f'更新图片失败: {str(e)}'}), 500

# 获取新闻列表API
@app.route('/api/news', methods=['GET', 'OPTIONS'])
def get_news_list():
    """获取新闻列表，支持按主题过滤"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Methods', 'GET')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        return response
    
    try:
        # 获取查询参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        topic = request.args.get('topic', '', type=str)
        
        # 构建查询
        query = News.query
        
        # 按主题过滤
        if topic and topic != 'all':
            query = query.filter(News.topic == topic)
        
        # 按上传时间倒序排序
        query = query.order_by(News.upload_time.desc())
        
        # 分页
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        news_items = pagination.items
        
        # 构建响应
        result = []
        for news in news_items:
            result.append({
                'id': news.id,
                'title': news.title,
                'content': news.content[:200] + '...' if len(news.content) > 200 else news.content,
                'topic': news.topic,
                'sentiment': news.sentiment,
                'upload_time': news.upload_time.isoformat()
            })
        
        return jsonify({
            'news': result,
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page
        })
    
    except Exception as e:
        app.logger.error(f"获取新闻列表时出错: {e}")
        return jsonify({'error': f'获取新闻列表失败: {str(e)}'}), 500

# 获取新闻详情API
@app.route('/api/news/<int:news_id>', methods=['GET', 'OPTIONS'])
def get_news_detail(news_id):
    """获取单个新闻详情"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Methods', 'GET')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        return response
    
    try:
        # 查询新闻
        news = News.query.get(news_id)
        if not news:
            return jsonify({'error': '新闻不存在'}), 404
        
        # 构建响应
        result = {
            'id': news.id,
            'title': news.title,
            'content': news.content,
            'topic': news.topic,
            'sentiment': news.sentiment,
            'upload_time': news.upload_time.isoformat()
        }
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"获取新闻详情时出错: {e}")
        return jsonify({'error': f'获取新闻详情失败: {str(e)}'}), 500

# 获取新闻主题列表API
@app.route('/api/news/topics', methods=['GET', 'OPTIONS'])
def get_news_topics():
    """获取所有新闻主题及每个主题的数量"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Methods', 'GET')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        return response
    
    try:
        # 查询所有可用主题及对应的新闻数量
        topics_with_count = db.session.query(
            News.topic, 
            db.func.count(News.id).label('count')
        ).group_by(News.topic).all()
        
        # 所有固定主题列表
        all_topics = [
            '体育', '娱乐', '科技', '时政', '财经', 
            '社会', '国际', '军事', '教育', '生活', '时尚'
        ]
        
        # 构建结果
        result = []
        
        # 添加"全部"选项
        total_count = News.query.count()
        result.append({
            'topic': 'all',
            'label': '全部',
            'count': total_count
        })
        
        # 处理实际数据库中的主题
        topic_dict = {topic: count for topic, count in topics_with_count}
        
        # 确保所有预定义主题都在结果中
        for topic in all_topics:
            count = topic_dict.get(topic, 0)
            if count > 0:  # 只显示有新闻的主题
                result.append({
                    'topic': topic,
                    'label': topic,
                    'count': count
                })
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"获取新闻主题列表时出错: {e}")
        return jsonify({'error': f'获取新闻主题列表失败: {str(e)}'}), 500

# 添加新闻上传接口
@app.route('/api/news/upload', methods=['POST', 'OPTIONS'])
@jwt_required(optional=True)
def upload_news():
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    # 获取用户ID
    current_user_id = None
    try:
        current_user_id = get_jwt_identity()
    except:
        pass  # 继续处理即使未登录
    
    # 获取请求数据
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "无效的请求数据"}), 400
    
    title = data.get('title', '')
    content = data.get('content', '')
    
    # 验证必填字段
    if not content:
        return jsonify({"status": "error", "message": "新闻内容是必填项"}), 400
    
    try:
        # 使用NewsAnalyzer分析新闻情感和主题
        analyzer = NewsAnalyzer()
        analysis_result = analyzer.analyze_and_save_news(title, content, db.session, News)
        
        return jsonify({
            "status": "success",
            "message": "新闻上传成功",
            "data": analysis_result
        }), 201
    
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"上传新闻时出错: {str(e)}")
        return jsonify({"status": "error", "message": f"上传新闻时出错: {str(e)}"}), 500

# 添加批量上传新闻接口
@app.route('/api/news/batch-upload', methods=['POST', 'OPTIONS'])
@jwt_required(optional=True)
def batch_upload_news():
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    # 获取用户ID（可选）
    current_user_id = None
    try:
        current_user_id = get_jwt_identity()
    except:
        pass  # 继续处理即使未登录
    
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "没有上传文件"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "未选择文件"}), 400
    
    # 检查文件格式
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.csv', '.xlsx', '.xls']:
        return jsonify({"status": "error", "message": "仅支持CSV和Excel格式文件"}), 400
    
    try:
        # 保存临时文件
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(temp_filepath)
        
        # 读取文件内容
        news_data = []
        if file_ext == '.csv':
            import csv
            with open(temp_filepath, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    news_data.append(row)
        else:  # Excel文件
            import pandas as pd
            df = pd.read_excel(temp_filepath)
            news_data = df.to_dict('records')
        
        # 初始化分析器
        analyzer = NewsAnalyzer()
        results = []
        success_count = 0
        
        # 处理每条新闻
        for i, news in enumerate(news_data):
            # 获取标题和内容
            title = news.get('title', '') or news.get('标题', '')
            content = news.get('content', '') or news.get('内容', '') or news.get('正文', '')
            
            # 跳过空内容
            if not content:
                results.append({
                    "index": i,
                    "status": "skipped",
                    "message": "新闻内容为空"
                })
                continue
            
            # 分析并保存新闻
            try:
                analysis_result = analyzer.analyze_and_save_news(title, content, db.session, News)
                if analysis_result.get("save_success", False):
                    success_count += 1
                results.append({
                    "index": i,
                    "status": "success" if analysis_result.get("save_success", False) else "error",
                    "data": analysis_result
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "status": "error",
                    "message": str(e)
                })
        
        # 删除临时文件
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        
        return jsonify({
            "status": "success",
            "message": f"批量上传完成，成功导入 {success_count}/{len(news_data)} 条新闻",
            "results": results
        }), 200
    
    except Exception as e:
        db.session.rollback()
        # 删除临时文件
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        
        app.logger.error(f"批量上传新闻时出错: {str(e)}")
        return jsonify({"status": "error", "message": f"批量上传新闻时出错: {str(e)}"}), 500


# 添加批量删除新闻接口
@app.route('/api/news/batch-delete', methods=['POST', 'OPTIONS'])
@jwt_required()
def batch_delete_news():
    """批量删除多条新闻"""
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
    
    # 仅允许管理员删除新闻
    if not current_user.is_admin:
        return jsonify({'error': '只有管理员可以删除新闻'}), 403
    
    # 获取请求数据
    data = request.get_json()
    if not data or not data.get('news_ids') or not isinstance(data.get('news_ids'), list):
        return jsonify({'error': '请提供要删除的新闻ID列表'}), 400
    
    news_ids = data.get('news_ids')
    
    # 统计结果
    result = {
        'success': 0,
        'failed': 0,
        'errors': []
    }
    
    try:
        for news_id in news_ids:
            # 查找要删除的新闻
            news = News.query.get(news_id)
            if not news:
                result['failed'] += 1
                result['errors'].append(f"新闻 {news_id} 不存在")
                continue
            
            try:
                # 从数据库中删除新闻记录
                db.session.delete(news)
                result['success'] += 1
            except Exception as e:
                result['failed'] += 1
                result['errors'].append(f"删除新闻 {news_id} 时出错: {str(e)}")
                continue
        
        # 提交所有更改
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'message': f'成功删除 {result["success"]} 条新闻，失败 {result["failed"]} 条',
            'details': result
        })
    
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"批量删除新闻时出错: {e}")
        return jsonify({'error': f'批量删除新闻失败: {str(e)}'}), 500


# 在应用启动时加载数据
def init_application_data():
    """初始化应用程序数据"""
    with app.app_context():
        # 确保所有表都被创建
        db.create_all()
        
        # 创建管理员账户
        try:
            # 检查admin账户是否存在
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                # 如果不存在，创建admin账户
                hashed_password = generate_password_hash('admin')
                admin_user = User(
                    username='admin',
                    password=hashed_password,
                    email='admin@example.com',
                    role='admin'
                )
                db.session.add(admin_user)
                db.session.commit()
                print("已创建管理员账户，用户名: admin，密码: admin")
            elif admin.role != 'admin':
                # 如果存在但不是管理员，更新为管理员
                admin.role = 'admin'
                db.session.commit()
                print("已将用户 admin 更新为管理员角色")
            else:
                print("管理员账户已存在")
        except Exception as e:
            print(f"创建管理员账户时出错: {e}")
        
        # 迁移图片URL
        migrate_image_urls()
        
        # 初始化图片数据库
        image_count = Image.query.count()
        if image_count == 0:
            print("数据库中没有图片记录，开始从CNA_images导入...")
            init_image_db_from_cna()
        else:
            print(f"数据库中已有 {image_count} 张图片记录")
            # 加载嵌入向量
            load_embeddings_from_db()
        
        # 初始化个性化推荐系统
        try:
            from RecSys import init_recommender
            init_recommender(app)
            print("个性化推荐系统初始化完成")
        except Exception as e:
            print(f"初始化推荐系统失败: {e}")

# 添加获取当前权重的API
@app.route('/api/weights', methods=['GET', 'OPTIONS'])
def get_weights():
    """获取当前推荐系统的权重配置"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Methods', 'GET')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        return response
    
    # 获取当前权重
    current_weights = get_current_weights()
    
    return jsonify({
        'success': True,
        'weights': current_weights
    })

# 添加修改权重的API（仅管理员可用）
@app.route('/api/weights', methods=['PUT', 'OPTIONS'])
@jwt_required()
def update_weights():
    """更新推荐系统的权重配置（仅管理员可用）"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Methods', 'PUT')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        return response
    
    # 验证用户身份和管理员权限
    current_user_id = get_jwt_identity()
    current_user = User.query.get(current_user_id)
    
    if not current_user:
        return jsonify({'error': '无效的用户身份'}), 401
        
    if not current_user.is_admin:
        return jsonify({'error': '只有管理员才能更新权重配置'}), 403
    
    # 获取请求数据
    data = request.get_json()
    if not data or not isinstance(data.get('weights'), dict):
        return jsonify({'error': '无效的权重数据'}), 400
    
    # 提取权重数据
    new_weights = data.get('weights')
    
    # 验证权重字段
    required_fields = [
        "标题", "事件内容", "动作内容", "实体内容", 
        "场景内容", "情感内容", "隐喻内容", "数据内容"
    ]
    
    for field in required_fields:
        if field not in new_weights:
            return jsonify({'error': f'缺少必要的权重字段: {field}'}), 400
        try:
            new_weights[field] = float(new_weights[field])
            if new_weights[field] < 0:
                return jsonify({'error': f'权重值不能为负: {field}'}), 400
        except:
            return jsonify({'error': f'权重值必须是数字: {field}'}), 400
    
    # 更新权重
    if set_weights(new_weights, is_admin=True):
        return jsonify({
            'success': True,
            'message': '权重配置已更新',
            'weights': get_current_weights()
        })
    else:
        return jsonify({'error': '更新权重配置失败'}), 500

# 添加获取文章推荐权重的API
@app.route('/api/article-weights', methods=['GET', 'OPTIONS'])
def get_article_weights_api():
    """获取当前文章推荐系统的权重配置"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Methods', 'GET')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        return response
    
    # 获取当前权重
    current_weights = get_article_weights()
    
    return jsonify({
        'success': True,
        'weights': current_weights
    })

# 添加修改文章推荐权重的API（仅管理员可用）
@app.route('/api/article-weights', methods=['PUT', 'OPTIONS'])
@jwt_required()
def update_article_weights():
    """更新文章推荐系统的权重配置（仅管理员可用）"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Methods', 'PUT')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        return response
    
    # 验证用户身份和管理员权限
    current_user_id = get_jwt_identity()
    current_user = User.query.get(current_user_id)
    
    if not current_user:
        return jsonify({'error': '无效的用户身份'}), 401
        
    if not current_user.is_admin:
        return jsonify({'error': '只有管理员才能更新权重配置'}), 403
    
    # 获取请求数据
    data = request.get_json()
    if not data or not isinstance(data.get('weights'), dict):
        return jsonify({'error': '无效的权重数据'}), 400
    
    # 提取权重数据
    new_weights = data.get('weights')
    
    # 验证权重字段
    required_fields = [
        "content_similarity", "theme_match", "sentiment_match"
    ]
    
    for field in required_fields:
        if field not in new_weights:
            return jsonify({'error': f'缺少必要的权重字段: {field}'}), 400
        try:
            new_weights[field] = float(new_weights[field])
            if new_weights[field] < 0:
                return jsonify({'error': f'权重值不能为负: {field}'}), 400
        except:
            return jsonify({'error': f'权重值必须是数字: {field}'}), 400
    
    # 确保权重总和为1
    total = sum(new_weights.values())
    if abs(total - 1.0) > 0.01:
        # 自动归一化
        new_weights = {k: v/total for k, v in new_weights.items()}
    
    # 更新权重
    if set_article_weights(new_weights, is_admin=True):
        return jsonify({
            'success': True,
            'message': '文章推荐权重配置已更新',
            'weights': get_article_weights()
        })
    else:
        return jsonify({'error': '更新文章推荐权重配置失败'}), 500

# 添加获取时间范围的API
@app.route('/api/article-time-range', methods=['GET', 'OPTIONS'])
def get_article_time_range_api():
    """获取当前文章推荐时间范围配置"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Methods', 'GET')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        return response
    
    # 获取当前时间范围
    current_time_range = get_time_range()
    
    return jsonify({
        'success': True,
        'time_range': current_time_range
    })

# 添加修改时间范围的API（仅管理员可用）
@app.route('/api/article-time-range', methods=['PUT', 'OPTIONS'])
@jwt_required()
def update_article_time_range():
    """更新文章推荐系统的时间范围配置（仅管理员可用）"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Methods', 'PUT')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        return response
    
    # 验证用户身份和管理员权限
    current_user_id = get_jwt_identity()
    current_user = User.query.get(current_user_id)
    
    if not current_user:
        return jsonify({'error': '无效的用户身份'}), 401
        
    if not current_user.is_admin:
        return jsonify({'error': '只有管理员才能更新时间范围配置'}), 403
    
    # 获取请求数据
    data = request.get_json()
    if not data or 'days' not in data:
        return jsonify({'error': '无效的时间范围数据'}), 400
    
    # 提取时间范围数据
    try:
        days = int(data['days'])
        if days < 0:
            days = 0  # 确保天数为非负整数
    except (ValueError, TypeError):
        return jsonify({'error': '天数必须是整数'}), 400
    
    # 更新时间范围
    new_time_range = {"days": days}
    if set_time_range(new_time_range, is_admin=True):
        return jsonify({
            'success': True,
            'message': '时间范围配置已更新',
            'time_range': get_time_range()
        })
    else:
        return jsonify({'error': '更新时间范围配置失败'}), 500

@app.route('/api/batch-upload', methods=['POST', 'OPTIONS'])
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
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            file.save(file_path)
            
            # 生成图片URL
            file_url = f"http://{Server_IP}/uploads/{new_filename}"
            
            # 计算图片嵌入向量
            try:
                image = cn_preprocess(PIL.Image.open(file_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = cn_clip_model.encode_image(image).cpu().numpy()[0]
                embedding_bytes = pickle.dumps(embedding)
            except Exception as e:
                print(f"计算嵌入向量失败: {e}")
                embedding_bytes = None
            
            # 创建图片记录
            new_image = Image(
                id=file_id,
                title=title,
                file_path=file_path,
                url=file_url,
                original_url=file_url,
                embedding=embedding_bytes,
                user_id=current_user_id  # 设置用户ID
            )
            db.session.add(new_image)
            
            # 添加空描述
            caption = Caption(caption="")
            new_image.captions.append(caption)
            
            # 记录成功
            results['success'] += 1
            results['images'].append({
                'id': file_id,
                'title': title,
                'url': file_url,
                'original_url': file_url
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
    
    # 返回处理结果
    return jsonify({
        'success': True,
        'message': f'批量上传完成: 成功 {results["success"]} 张，失败 {results["failed"]} 张',
        'results': results
    }), 201


if __name__ == '__main__':
    print("后端服务启动中...")
    init_application_data()
    app.run(debug=False)