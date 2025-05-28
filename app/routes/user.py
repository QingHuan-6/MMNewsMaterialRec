from flask import jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required, create_access_token, verify_jwt_in_request
from werkzeug.security import generate_password_hash, check_password_hash

from ..models.models import Download, db
from ..models.models import User, Image, Tag, ImageTag, Caption, Favorite, SearchHistory, ViewHistory, News
from ..models.models import db
from . import user_bp


# 用户注册
@user_bp.route('/api/register', methods=['POST'])
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
@user_bp.route('/api/login', methods=['POST'])
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
@user_bp.route('/api/user', methods=['GET'])
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
@user_bp.route('/api/favorites', methods=['POST', 'OPTIONS'])
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

# 修改获取收藏列表
@user_bp.route('/api/favorites', methods=['GET', 'OPTIONS'])
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
@user_bp.route('/api/favorites/<image_id>', methods=['DELETE', 'OPTIONS'])
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
@user_bp.route('/api/downloads', methods=['POST', 'OPTIONS'])
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
@user_bp.route('/api/downloads', methods=['GET', 'OPTIONS'])
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
@user_bp.route('/api/downloads', methods=['DELETE', 'OPTIONS'])
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


@user_bp.route('/api/search_history', methods=['POST', 'OPTIONS'])
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
@user_bp.route('/api/view_history', methods=['POST', 'OPTIONS'])
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
@user_bp.route('/api/search_history', methods=['GET', 'OPTIONS'])
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
@user_bp.route('/api/view_history', methods=['GET', 'OPTIONS'])
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
