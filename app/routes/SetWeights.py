from flask import jsonify, request,blueprints
from flask_jwt_extended import get_jwt_identity, jwt_required
from ..services.novo1 import set_weights, get_current_weights
from ..services.novo2 import set_time_range, get_time_range, set_article_weights, get_article_weights
from ..models.models import User
from . import set_weights_bp

# 添加获取当前权重
@set_weights_bp.route('/api/weights', methods=['GET', 'OPTIONS'])
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

# 添加修改权重
@set_weights_bp.route('/api/weights', methods=['PUT', 'OPTIONS'])
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

# 添加获取文章推荐权重
@set_weights_bp.route('/api/article-weights', methods=['GET', 'OPTIONS'])
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

# 添加修改文章推荐权重
@set_weights_bp.route('/api/article-weights', methods=['PUT', 'OPTIONS'])
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
        new_weights = {k: v/ total for k, v in new_weights.items()}

    # 更新权重
    if set_article_weights(new_weights, is_admin=True):
        return jsonify({
            'success': True,
            'message': '文章推荐权重配置已更新',
            'weights': get_article_weights()
        })
    else:
        return jsonify({'error': '更新文章推荐权重配置失败'}), 500


# 添加获取时间范围
@set_weights_bp.route('/api/article-time-range', methods=['GET', 'OPTIONS'])
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


# 添加修改时间范围
@set_weights_bp.route('/api/article-time-range', methods=['PUT', 'OPTIONS'])
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
