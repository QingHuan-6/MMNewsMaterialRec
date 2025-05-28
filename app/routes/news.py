import os
from flask import jsonify, request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from ..models.models import News, db, User
from ..services.novo1 import NewsAnalyzer, ImageRecommender
from . import news_bp
import pickle
import numpy as np

# 导入 Elasticsearch 工具函数
from ..utils.es_utils import index_image_vector, bulk_index_vectors, es_client, NEWS_INDEX_NAME



# 获取新闻列表
@news_bp.route('/api/news', methods=['GET', 'OPTIONS'])
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
        current_app.logger.error(f"获取新闻列表时出错: {e}")
        return jsonify({'error': f'获取新闻列表失败: {str(e)}'}), 500

# 获取新闻详情
@news_bp.route('/api/news/<int:news_id>', methods=['GET', 'OPTIONS'])
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
        current_app.logger.error(f"获取新闻详情时出错: {e}")
        return jsonify({'error': f'获取新闻详情失败: {str(e)}'}), 500

# 获取新闻主题列表
@news_bp.route('/api/news/topics', methods=['GET', 'OPTIONS'])
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
        current_app.logger.error(f"获取新闻主题列表时出错: {e}")
        return jsonify({'error': f'获取新闻主题列表失败: {str(e)}'}), 500

# 添加新闻上传接口
@news_bp.route('/api/news/upload', methods=['POST', 'OPTIONS'])
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
        current_app.logger.error(f"上传新闻时出错: {str(e)}")
        return jsonify({"status": "error", "message": f"上传新闻时出错: {str(e)}"}), 500

# 添加批量上传新闻接口
@news_bp.route('/api/news/batch-upload', methods=['POST', 'OPTIONS'])
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
        temp_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
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
        
        # 用于批量索引到 Elasticsearch 的数据
        es_vectors_data = []

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
                    
                    # 查找刚保存的新闻记录，获取其ID和向量
                    news_id = analysis_result.get("id")
                    if news_id:
                        saved_news = News.query.get(news_id)
                        if saved_news and saved_news.embedding:
                            try:
                                embedding = pickle.loads(saved_news.embedding)
                                # 准备 Elasticsearch 索引数据
                                es_vectors_data.append({
                                    'image_id': f"news_{news_id}",
                                    'news_id': news_id,
                                    'vector': embedding,
                                    'title': title,
                                    'content': content[:50],
                                    'topic': saved_news.topic,
                                    'sentiment': saved_news.sentiment
                                })
                            except Exception as e:
                                current_app.logger.error(f"处理新闻 {news_id} 向量时出错: {str(e)}")
                    
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

        # 批量索引到 Elasticsearch
        if es_vectors_data:
            try:
                bulk_index_vectors(es_vectors_data)
                current_app.logger.info(f"成功将 {len(es_vectors_data)} 条新闻索引到 Elasticsearch")
            except Exception as e:
                current_app.logger.error(f"批量索引新闻到 Elasticsearch 失败: {str(e)}")

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

        current_app.logger.error(f"批量上传新闻时出错: {str(e)}")
        return jsonify({"status": "error", "message": f"批量上传新闻时出错: {str(e)}"}), 500


# 添加批量删除新闻接口
@news_bp.route('/api/news/batch-delete', methods=['POST', 'OPTIONS'])
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
                # 从 Elasticsearch 中删除新闻向量
                try:
                    if es_client:
                        es_client.delete(index=NEWS_INDEX_NAME, id=f"news_{news_id}", ignore=[404])
                        current_app.logger.info(f"已从 Elasticsearch 中删除新闻 {news_id}")
                except Exception as e:
                    current_app.logger.error(f"从 Elasticsearch 删除新闻 {news_id} 时出错: {str(e)}")

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
        current_app.logger.error(f"批量删除新闻时出错: {e}")
        return jsonify({'error': f'批量删除新闻失败: {str(e)}'}), 500

# 添加新闻向量迁移到 Elasticsearch 的接口
@news_bp.route('/api/news/migrate-vectors', methods=['POST', 'OPTIONS'])
def migrate_news_vectors():
    """将现有的新闻向量数据迁移到 Elasticsearch"""
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        return response

    try:
        # 获取所有带有向量的新闻
        news_items = News.query.filter(News.embedding.isnot(None)).all()
        
        if not news_items:
            return jsonify({'message': '没有找到带有向量的新闻数据'}), 404
        
        current_app.logger.info(f"找到 {len(news_items)} 条带有向量的新闻")
        
        # 准备批量索引数据
        vectors_data = []
        skipped_count = 0
        
        for news in news_items:
            try:
                if news.embedding:
                    # 解析向量数据
                    embedding = pickle.loads(news.embedding)
                    
                    # 验证向量数据
                    if not isinstance(embedding, (list, np.ndarray)):
                        current_app.logger.warning(f"新闻 {news.id} 的向量数据类型不正确: {type(embedding)}")
                        skipped_count += 1
                        continue
                    
                    # 确保向量维度正确
                    if isinstance(embedding, np.ndarray) and embedding.shape[0] != 512:
                        current_app.logger.warning(f"新闻 {news.id} 的向量维度不正确: {embedding.shape}")
                        skipped_count += 1
                        continue
                    
                    # 准备元数据
                    metadata = {
                        'news_id': news.id,
                        'title': news.title if news.title else "",
                        'content': news.content[:50] if news.content else "",
                        'topic': news.topic if news.topic else "",
                        'sentiment': news.sentiment if news.sentiment else "",
                        'upload_time': news.upload_time.isoformat() if news.upload_time else ""
                    }
                    
                    # 添加到批量索引列表
                    vectors_data.append({
                        'image_id': f"news_{news.id}",
                        'vector': embedding,
                        **metadata
                    })
            except Exception as e:
                current_app.logger.error(f"处理新闻 {news.id} 的向量数据时出错: {str(e)}")
                skipped_count += 1
                continue
        
        current_app.logger.info(f"准备索引 {len(vectors_data)} 条新闻向量，跳过 {skipped_count} 条")
        
        # 批量索引到 Elasticsearch
        if vectors_data:
            # 分批处理，每批最多100条
            batch_size = 50
            total_success = 0
            total_failed = 0
            
            for i in range(0, len(vectors_data), batch_size):
                batch = vectors_data[i:i+batch_size]
                current_app.logger.info(f"处理批次 {i//batch_size + 1}/{(len(vectors_data)-1)//batch_size + 1}，包含 {len(batch)} 条数据")
                
                try:
                    bulk_index_vectors(batch)
                    total_success += len(batch)
                except Exception as e:
                    current_app.logger.error(f"批次 {i//batch_size + 1} 索引失败: {str(e)}")
                    total_failed += len(batch)
            
            return jsonify({
                'message': f'新闻向量迁移完成: 成功 {total_success} 条，失败 {total_failed} 条，跳过 {skipped_count} 条',
                'success_count': total_success,
                'failed_count': total_failed,
                'skipped_count': skipped_count,
                'total_count': len(news_items)
            })
        else:
            return jsonify({'message': '没有有效的向量数据可迁移'}), 404
            
    except Exception as e:
        current_app.logger.error(f"迁移新闻向量数据时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


