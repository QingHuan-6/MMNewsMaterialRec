import time
import numpy as np
from flask import jsonify, request,blueprints
from ..services.novo1 import ImageRecommender,NewsAnalyzer, get_current_weights, set_weights

from ..services.novo2 import recommend_articles
from ..models.models import db, News
from . import novo_bp



news_analyzer = NewsAnalyzer()
image_recommender = ImageRecommender()

@novo_bp.route('/api/novo/get_recommendations_news', methods=['POST', 'OPTIONS'])
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

@novo_bp.route('/api/novo/get_recommendations_images', methods=['POST', 'OPTIONS'])
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

        news_elements =news_analyzer.analyze_news(title ,content)
        image_recommendations =image_recommender.recommend_images(news_elements ,title ,content
                                                                   ,use_fixed_weights=True)
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
        recommendations ={
            "news_id" :"",
            "news_title" :"",
            "features" :features,
            "recommended_images" :[{
                "image_path" :img_url,
                "similarity_score" :float(score)
            }
                for img_url, score in image_recommendations
            ]
        }


        # 构建响应
        return jsonify(recommendations)

    except Exception as e:
        print(f"获取推荐图片失败: {str(e)}")
        return jsonify({"error": f"获取推荐图片失败: {str(e)}"}), 500

