import os
import torch
import numpy as np
from PIL import Image
import clip
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from ..models.models import db, User, Image as ImageModel, Favorite, Download, SearchHistory, ViewHistory
from flask import current_app, jsonify
import pickle
from ..utils.utils import load_cn_clip_model
# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_cn_clip_model()
model.to(device)
model.eval()

class RecommendationSystem:
    def __init__(self, app=None, cache_dir="cache"):
        """初始化推荐系统"""
        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.app = app
        
        # 设置缓存目录
        current_dir = os.getcwd()
        self.cache_dir = os.path.join(current_dir, cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 图片特征缓存
        self.image_features_cache = {}
    #从数据库加载最新图片特征
    def load_image_features_from_db(self, use_cache=True):
        """从数据库加载所有图片特征"""
        cache_file = os.path.join(self.cache_dir, "db_image_embeddings.npz")
        
        # 从缓存加载
        if use_cache and os.path.exists(cache_file):
            try:
                with np.load(cache_file) as data:
                    image_features = {
                        image_id: torch.from_numpy(arr).to(device).float()
                        for image_id, arr in data.items()
                    }
                print(f"[信息] 已从缓存加载 {len(image_features)} 张图片特征")
                return image_features
            except Exception as e:
                print(f"[警告] 缓存加载失败，将从数据库重新读取: {e}")
        
        # 确保有应用上下文
        if self.app is None:
            raise RuntimeError("未提供Flask应用实例，无法进行数据库操作")
        
        # 在应用上下文中执行数据库操作
        with self.app.app_context():
            # 从数据库加载
            all_images = ImageModel.query.all()
            image_features = {}
            
            for image in tqdm(all_images, desc="从数据库加载图片特征"):
                if image.embedding:
                    try:
                        # 使用pickle.loads解析二进制embedding数据
                        embedding = pickle.loads(image.embedding)
                        # 将embedding转换为torch张量
                        image_features[image.id] = torch.from_numpy(
                            np.array(embedding, dtype=np.float32)
                        ).to(device).float()
                    except Exception as e:
                        print(f"[错误] 无法处理图片 {image.id} 的特征: {e}")
        
        # 保存缓存
        if use_cache and image_features:
            np_features = {img_id: feat.cpu().numpy() for img_id, feat in image_features.items()}
            np.savez_compressed(cache_file, **np_features)
            print(f"[信息] 已保存 {len(image_features)} 张图片特征到缓存")
        
        return image_features
    #从文件夹加载图片特征
    def load_image_features_from_folder(self, image_folder, cache_name=None, use_cache=True):
        """从文件夹加载图片特征（用于推荐库）"""
        if cache_name is None:
            cache_name = f"folder_{hash(image_folder)}"
        
        cache_file = os.path.join(self.cache_dir, f"{cache_name}.npz")
        
        # 从NPZ缓存加载
        if use_cache and os.path.exists(cache_file):
            try:
                with np.load(cache_file) as data:
                    image_features = {
                        path: torch.from_numpy(arr).to(device).float()
                        for path, arr in data.items()
                    }
                print(f"[信息] 已从缓存加载 {len(image_features)} 张图片特征")
                return image_features
            except Exception as e:
                print(f"[警告] 缓存加载失败，将重新计算特征: {e}")
        
        # 实时计算特征
        image_features = {}
        file_list = [
            os.path.join(root, f) 
            for root, _, fs in os.walk(image_folder)
            for f in fs if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        for image_path in tqdm(file_list, desc=f"计算 {image_folder} 图片特征"):
            try:
                image = Image.open(image_path).convert("RGB")
                image_input = self.preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    feature = self.model.encode_image(image_input)
                image_features[image_path] = feature
            except Exception as e:
                print(f"[错误] 跳过损坏图片 {image_path}: {e}")
        
        # 保存为NPZ缓存
        if use_cache and image_features:
            np_features = {img_id: feat.cpu().numpy() for img_id, feat in image_features.items()}
            np.savez_compressed(cache_file, **np_features)
            print(f"[信息] 已保存 {len(image_features)} 张图片特征到缓存")
        
        return image_features
    #计算时间衰减权重
    def calculate_time_decay(self, timestamp, half_life_days=1):
        """计算时间衰减权重，使用半衰期模型"""
        if not timestamp:
            return 0.0
        
        try:
            # 计算时间差（天数）
            current_time = datetime.now()
            time_diff = current_time - timestamp
            days_diff = time_diff.total_seconds() / (24 * 3600)
            
            # 半衰期衰减: weight = 2^(-t/half_life)
            weight = 2 ** (-days_diff / half_life_days)
            return weight
        except Exception as e:
            print(f"[警告] 计算时间权重失败: {e}")
            return 0.0
    
    #获取用户检索记录的特征向量
    def get_user_search_features(self, user_id, max_records=20):
        """从数据库获取用户检索记录的特征向量"""
        if self.app is None:
            raise RuntimeError("未提供Flask应用实例，无法进行数据库操作")
            
        try:
            with self.app.app_context():
                # 查询用户最近的检索记录，按时间降序排列
                search_records = SearchHistory.query.filter_by(user_id=user_id)\
                    .order_by(SearchHistory.added_at.desc())\
                    .limit(max_records).all()
                
                if not search_records:
                    return None, 0
                
                search_texts = []
                time_weights = []
                
                for record in search_records:
                    search_texts.append(record.search_content)
                    # 计算时间权重
                    time_weight = self.calculate_time_decay(record.added_at)
                    time_weights.append(time_weight)
            
            # 计算总权重
            total_weight = sum(time_weights) if time_weights else 1
            normalized_weights = [w/total_weight for w in time_weights]
            
            # 使用CLIP模型编码文本
            text_inputs = clip.tokenize(search_texts).to(device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_inputs)
            
            # 应用时间权重
            weighted_features = torch.zeros_like(text_features[0]).unsqueeze(0)
            for i, weight in enumerate(normalized_weights):
                weighted_features += text_features[i].unsqueeze(0) * weight
            
            # 归一化
            weighted_features = weighted_features / weighted_features.norm(dim=-1, keepdim=True)
            
            return weighted_features, 0.5  # 0.5是检索记录在整体推荐中的基础权重
        except Exception as e:
            print(f"[错误] 获取用户检索特征失败: {e}")
            return None, 0
    #获取用户收藏图片的特征向量
    def get_user_favorites_features(self, user_id, image_features):
        """从数据库获取用户收藏图片的特征向量"""
        if self.app is None:
            raise RuntimeError("未提供Flask应用实例，无法进行数据库操作")
            
        try:
            with self.app.app_context():
                # 查询用户的收藏记录，按时间降序排列
                favorites = Favorite.query.filter_by(user_id=user_id)\
                    .order_by(Favorite.added_at.desc()).all()
                
                if not favorites:
                    return None, 0
                
                favorite_features = []
                time_weights = []
                
                for favorite in favorites:
                    if favorite.image_id in image_features:
                        favorite_features.append(image_features[favorite.image_id])
                        # 计算时间权重
                        time_weight = self.calculate_time_decay(favorite.added_at)
                        time_weights.append(time_weight)
            
            if not favorite_features:
                return None, 0
            
            # 计算总权重
            total_weight = sum(time_weights) if time_weights else 1
            normalized_weights = [w/total_weight for w in time_weights]
            
            # 应用时间权重
            weighted_features = torch.zeros_like(favorite_features[0])
            for i, weight in enumerate(normalized_weights):
                weighted_features += favorite_features[i] * weight
            
            # 归一化
            weighted_features = weighted_features / weighted_features.norm(dim=-1, keepdim=True)
            
            return weighted_features, 0.3  # 0.3是收藏记录在整体推荐中的基础权重
        except Exception as e:
            print(f"[错误] 获取用户收藏特征失败: {e}")
            return None, 0
    #获取用户浏览记录的特征向量
    def get_user_view_features(self, user_id, image_features):
        """从数据库获取用户浏览记录的特征向量"""
        if self.app is None:
            raise RuntimeError("未提供Flask应用实例，无法进行数据库操作")
            
        try:
            with self.app.app_context():
                # 查询用户的浏览记录，按时间降序排列
                views = ViewHistory.query.filter_by(user_id=user_id)\
                    .order_by(ViewHistory.added_at.desc()).all()
                
                if not views:
                    return None, 0
                
                view_features = []
                time_weights = []
                
                for view in views:
                    if view.image_id in image_features:
                        view_features.append(image_features[view.image_id])
                        # 计算时间权重
                        time_weight = self.calculate_time_decay(view.added_at)
                        time_weights.append(time_weight)
            
            if not view_features:
                return None, 0
            
            # 计算总权重
            total_weight = sum(time_weights) if time_weights else 1
            normalized_weights = [w/total_weight for w in time_weights]
            
            # 应用时间权重
            weighted_features = torch.zeros_like(view_features[0])
            for i, weight in enumerate(normalized_weights):
                weighted_features += view_features[i] * weight
            
            # 归一化
            weighted_features = weighted_features / weighted_features.norm(dim=-1, keepdim=True)
            
            return weighted_features, 0.2  # 0.2是浏览记录在整体推荐中的基础权重
        except Exception as e:
            print(f"[错误] 获取用户浏览特征失败: {e}")
            return None, 0
    
    #生成个性化推荐
    def get_recommendations(self, user_id, top_n=30, use_cache=True):
        """为指定用户生成个性化推荐"""
        if self.app is None:
            raise RuntimeError("未提供Flask应用实例，无法进行数据库操作")
            
        start_time = time.time()
        print(f"[信息] 开始为用户 {user_id} 生成推荐...")
        
        # 加载所有图片特征
        image_features = self.load_image_features_from_db(use_cache=use_cache)
        if not image_features:
            print("[错误] 无法加载图片特征")
            return []
        
        # 收集用户偏好向量
        weighted_vectors = []
        
        # 获取用户搜索记录特征
        search_feature, search_weight = self.get_user_search_features(user_id)
        if search_feature is not None:
            weighted_vectors.append((search_feature, search_weight))
        
        # 获取用户收藏记录特征
        fav_feature, fav_weight = self.get_user_favorites_features(user_id, image_features)
        if fav_feature is not None:
            weighted_vectors.append((fav_feature, fav_weight))
        
        # 获取用户浏览记录特征
        view_feature, view_weight = self.get_user_view_features(user_id, image_features)
        if view_feature is not None:
            weighted_vectors.append((view_feature, view_weight))
        
        # 如果没有任何用户数据，返回空列表
        if not weighted_vectors:
            print(f"[警告] 用户 {user_id} 没有任何偏好数据")
            return []
        
        # 融合用户偏好特征
        total_weight = sum(w for _, w in weighted_vectors)
        fused_feature = sum(f * (w / total_weight) for f, w in weighted_vectors)
        fused_feature = fused_feature / fused_feature.norm(dim=-1, keepdim=True)
        
        # 计算相似度并排序
        image_ids = list(image_features.keys())
        image_vectors = torch.stack(list(image_features.values())).to(device)
        
        with torch.no_grad():
            similarities = (fused_feature @ image_vectors.mT).squeeze().cpu().tolist()
        
        # 按相似度降序排序
        sorted_items = sorted(zip(similarities, image_ids), key=lambda x: -x[0])
        
        with self.app.app_context():
            # 过滤已经浏览和收藏过的项目
            filtered_items = []
            seen_ids = set()
            
            # 获取用户已浏览和收藏的图片ID
            user_favorites = {f.image_id for f in Favorite.query.filter_by(user_id=user_id).all()}
            user_views = {v.image_id for v in ViewHistory.query.filter_by(user_id=user_id).all()}
            
            # 首先添加一些用户没看过的图片
            for sim, img_id in sorted_items:
                if img_id not in user_views and img_id not in user_favorites and img_id not in seen_ids:
                    seen_ids.add(img_id)
                    filtered_items.append((sim, img_id))
                    if len(filtered_items) >= top_n * 0.7:  # 70%是新内容
                        break
            
            # 添加一些用户看过但没收藏的图片(最多20%)
            viewed_not_favorited = user_views - user_favorites
            count_to_add = int(top_n * 0.2)
            for sim, img_id in sorted_items:
                if img_id in viewed_not_favorited and img_id not in seen_ids:
                    seen_ids.add(img_id)
                    filtered_items.append((sim, img_id))
                    count_to_add -= 1
                    if count_to_add <= 0:
                        break
            
            # 添加一些用户收藏过的图片(最多10%)
            count_to_add = int(top_n * 0.1)
            for sim, img_id in sorted_items:
                if img_id in user_favorites and img_id not in seen_ids:
                    seen_ids.add(img_id)
                    filtered_items.append((sim, img_id))
                    count_to_add -= 1
                    if count_to_add <= 0:
                        break
        
        # 按相似度重新排序
        recommendations = sorted(filtered_items, key=lambda x: -x[0])
        
        # 提取图片ID列表
        result_ids = [img_id for _, img_id in recommendations[:top_n]]
        
        elapsed = time.time() - start_time
        print(f"[信息] 用户 {user_id} 推荐完成，耗时 {elapsed:.2f} 秒，推荐 {len(result_ids)} 张图片")
        
        return result_ids


    def get_recommendation_details(self, user_id, top_n=30, use_cache=True):
        """获取推荐图片的完整详细信息"""
        result_ids = self.get_recommendations(user_id, top_n, use_cache)
        
        with self.app.app_context():
            # 获取每个推荐图片的详细信息
            recommendations = []
            
            for img_id in result_ids:
                # 从数据库获取图片信息
                image = ImageModel.query.get(img_id)
                
                if image:
                    # 获取图片的描述和标签
                    captions_list = [c.caption for c in image.captions]
                    
                    recommendations.append({
                        'id': image.id,
                        'url': image.thumbnail_url,
                        'original_url': image.thumbnail_url,
                        'title': image.title,
                        'captions': captions_list,
                        'tags': []
                    })
            
            return recommendations



# 创建推荐系统的单例实例(避免多次创建重复)
_recommender_instance = None

def get_recommender_instance(app):
    """获取或创建推荐系统的单例实例"""
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = RecommendationSystem(app=app)
        print("[信息] 已创建推荐系统实例")
    return _recommender_instance



# 将推荐系统集成到Flask应用
def init_recommender(app):
    """初始化推荐系统并注册路由"""
    # 创建推荐系统实例
    recommender = get_recommender_instance(app)
    
    # 注册获取推荐的API路由
    @app.route('/api/recommendations', methods=['GET'])
    def get_recommendations():
        from flask_jwt_extended import jwt_required, get_jwt_identity
        from flask import request
        
        # 要求用户登录
        @jwt_required()
        def get_user_recommendations():
            try:
                # 获取用户ID
                user_id = get_jwt_identity()
                if not user_id:
                    return jsonify({'error': 'Authentication required'}), 401
                
                # 获取参数
                limit = request.args.get('limit', 30, type=int)
                refresh = request.args.get('refresh', 'false').lower() == 'true'
                
                # 获取推荐
                with app.app_context():
                    recommendations = recommender.get_recommendation_details(
                        user_id=user_id, 
                        top_n=limit,
                        use_cache=not refresh
                    )
                
                return jsonify({
                    'total': len(recommendations),
                    'recommendations': recommendations
                })
            except Exception as e:
                app.logger.error(f"获取推荐失败: {str(e)}")
                return jsonify({'error': f'获取推荐失败: {str(e)}'}), 500
        
        return get_user_recommendations()
    
    print("[信息] 已注册推荐系统API路由")


# 只在直接运行此文件时执行的测试代码
if __name__ == "__main__":
    from app import app  # 导入Flask应用实例
    
    # 初始化推荐系统
    init_recommender(app)
    
    # 创建推荐系统实例并传入Flask应用
    recommender = get_recommender_instance(app)
    
    # 假设我们要为用户ID为1的用户生成推荐
    user_id = 1
    
    with app.app_context():
        recommendations = recommender.get_recommendation_details(user_id, top_n=30)
    
    print(f"为用户 {user_id} 推荐的图片:")
    for i, img in enumerate(recommendations, 1):
        print(f"  {i}. {img['id']} - {img['title']}")