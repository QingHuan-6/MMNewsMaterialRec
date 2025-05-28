import time
import os
import json
import numpy as np
import pandas as pd
import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
import optuna
from PIL import Image
from tqdm import tqdm
from . import LLM
import pymysql
import pickle
import base64
from datetime import datetime, timezone, timedelta
from .novo2 import batch_get_keywords, batch_text_embedding, preprocess_text
from sklearn.metrics.pairwise import cosine_similarity
from flask import current_app

# 导入 Elasticsearch 工具函数
try:
    from ..utils.es_utils import index_image_vector, bulk_index_vectors, index_news_vector
except ImportError:
    # 如果在非应用上下文中使用，提供一个备用的函数
    def index_image_vector(image_id, vector, metadata):
        print(f"备用函数: 索引新闻向量到 Elasticsearch (ID: {image_id})")
        return True
    
    def bulk_index_vectors(vectors_data):
        print(f"备用函数: 批量索引向量到 Elasticsearch (数量: {len(vectors_data)})")
        return True

# 导入工具模块的CLIP加载函数
try:
    from ..utils.utils import load_cn_clip_model
except ImportError:
    # 如果在非应用上下文中使用，提供一个备用的加载函数
    def load_cn_clip_model():
        """备用的CN-CLIP模型加载函数"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "ViT-B-16"
        try:
            from cn_clip.clip import load_from_name
            model, preprocess = load_from_name(model_name, device=device)
            model.eval()
            print(f"CN-CLIP模型({model_name})直接加载成功")
            return model, preprocess
        except Exception as e:
            print(f"CN-CLIP模型加载失败: {e}")
            return None, None

from opencc import OpenCC
cc = OpenCC('t2s')

# 全局固定权重配置
DEFAULT_WEIGHTS = {
    "标题": 0.3,  # 标题通常包含核心信息，权重较高
    "事件内容": 0.25,  # 核心事件是新闻的核心语义单元
    "动作内容": 0.05,  # 关键动作描述事件动态
    "实体内容": 0.25,  # 显著实体（人名、机构名）是重要匹配线索
    "场景内容": 0.08,  # 场景特征（时间、地点）辅助定位
    "情感内容": 0.03,  # 情感倾向对视觉氛围有影响，但权重较低
    "隐喻内容": 0.02,  # 视觉隐喻较抽象，权重最低
    "数据内容": 0.02,  # 数据统计作为补充信息
}

# 确保权重和为1
total_weight = sum(DEFAULT_WEIGHTS.values())
GLOBAL_WEIGHTS = {k: v / total_weight for k, v in DEFAULT_WEIGHTS.items()}

# 权重配置文件路径
WEIGHTS_CONFIG_PATH = "weights_config.json"

# 加载已保存的权重配置（如果存在）
def load_weights_config():
    global GLOBAL_WEIGHTS
    try:
        if os.path.exists(WEIGHTS_CONFIG_PATH):
            with open(WEIGHTS_CONFIG_PATH, 'r', encoding='utf-8') as f:
                weights = json.load(f)
                # 验证权重配置是否完整
                if all(k in weights for k in DEFAULT_WEIGHTS.keys()):
                    # 确保权重和为1
                    total = sum(weights.values())
                    GLOBAL_WEIGHTS = {k: v / total for k, v in weights.items()}
                    print(f"已加载自定义权重配置: {GLOBAL_WEIGHTS}")
                    return GLOBAL_WEIGHTS
    except Exception as e:
        print(f"加载权重配置出错: {e}")
    
    print(f"使用默认权重配置: {GLOBAL_WEIGHTS}")
    return GLOBAL_WEIGHTS

# 保存权重配置
def save_weights_config(weights):
    try:
        with open(WEIGHTS_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(weights, f, ensure_ascii=False, indent=2)
        print(f"权重配置已保存到: {WEIGHTS_CONFIG_PATH}")
        return True
    except Exception as e:
        print(f"保存权重配置出错: {e}")
        return False

# 更新全局权重配置
def update_weights(new_weights):
    global GLOBAL_WEIGHTS
    # 验证权重配置是否完整
    if not all(k in new_weights for k in DEFAULT_WEIGHTS.keys()):
        print("权重配置不完整，更新失败")
        return False
    
    # 确保权重和为1
    total = sum(new_weights.values())
    GLOBAL_WEIGHTS = {k: v / total for k, v in new_weights.items()}
    
    # 保存到文件
    if save_weights_config(GLOBAL_WEIGHTS):
        return True
    return False

# 加载初始权重配置
load_weights_config()

class NewsAnalyzer:
    """新闻信息提取器类"""
    
    def __init__(self):
        """初始化新闻分析器"""
        self.llm_client = LLM.LLMClient()


    #分析新闻情感和主题    
    def analyze_news_sentiment_topic(self, content):
        """
        分析新闻情感和主题
        
        参数:
            content: 新闻正文
            
        返回:
            tuple: (情感, 主题)
        """
        # 定义情感分类和系统提示
        SENTIMENTS = ["负面", "中性", "正面"]
        
        # 增加示例和更详细的判断标准
        sentiment_prompt = f"""你是一个专业的新闻情感分析师。请根据新闻内容，分析新闻文章的情感（负面/中性/正面）。
        只输出情感类型，若新闻未提及任何主题相关内容。情感包括：{', '.join(SENTIMENTS)}。
        情感的判断如下：
        1. 正面：新闻内容若主要包含积极、肯定、赞扬的表达，则可判定为正面情感。例如："某公司成功研发出新型产品，市场反响良好。"
        2. 中性：新闻内容若只是客观陈述事实，不带有明显的情感偏向，即为中性情感。例如："今天的气温是25摄氏度。"
        3. 负面：若新闻内容包含消极、否定、批判的表达，则为负面情感。例如："某企业因违规操作被处罚。"
        请确保输出格式正确，无需其他解释。
        
        新闻内容为："{content}"
        请你分析新闻的情感。
        """
        
        # 调用LLM获取情感分析结果
        sentiment_result = self.llm_client.ask(sentiment_prompt)
        sentiment = sentiment_result["answer"].strip()
        
        # 后处理情感结果，确保得到有效输出
        if sentiment not in SENTIMENTS:
            # 增加更严格的后处理，若不匹配则尝试根据关键词判断
            for keyword in ["积极", "肯定", "赞扬"]:
                if keyword in content:
                    sentiment = "正面"
                    break
            else:
                for keyword in ["消极", "否定", "批判"]:
                    if keyword in content:
                        sentiment = "负面"
                        break
                else:
                    sentiment = "中性"
        
        # 定义主题分类和系统提示
        TOPICS = ["体育", "娱乐", "科技", "时政", "财经", "社会", "国际", "军事", "教育", "生活", "时尚"]
        
        # 主题分类提示
        topic_prompt = f"""你是一个专业的新闻分类专家。请根据新闻内容，从以下 11 个主题中选择一个合适的主题进行分类，一条新闻只能匹配一个主题。
        只输出主题名称，不要有其他多余表述。主题包括：{', '.join(TOPICS)}。
        主题的定义如下：
        1. 体育：报道竞技体育活动、体育产业发展及运动员相关动态的新闻。例如："某足球比赛精彩落幕，XX队获得冠军。"
        2. 娱乐：涉及影视演艺行业及公众人物非职业动态的报道。例如："某明星举办婚礼，众多好友出席。"
        3. 科技：关于技术创新及数字化领域发展的报道。例如："人工智能技术取得重大突破。"
        4. 时政：我国政府机构运作及国内政治生态相关报道。例如："政府出台新的税收政策。"
        5. 财经：宏观经济运行及商业金融领域动态。例如："股市今日大幅上涨。"
        6. 社会：民生问题及公共领域事件报道。例如："某社区开展志愿者活动。"
        7. 国际：非军事类跨国事务及外交动态。例如："两国签署贸易协议。"
        8. 军事：国防建设及武装力量相关动态。例如："某军区举行军事演习。"
        9. 教育：教育体系改革及学术发展动态。例如："某高校改革招生政策。"
        10. 生活：居民日常消费及基础民生服务。例如："某超市推出优惠活动。"
        11. 时尚：潮流趋势与美学设计领域动态。例如："某品牌发布新款时装。"
        请确保输出格式正确，无需其他解释。
        
        新闻内容为："{content}"
        请你进行主题分类。
        """
        
        # 调用LLM获取主题分析结果
        topic_result = self.llm_client.ask(topic_prompt)
        topic = topic_result["answer"].strip()
        
        # 后处理主题结果，确保得到有效输出
        if topic not in TOPICS:
            topic = "社会"  # 默认主题
        
        return sentiment, topic

    #保存新闻到数据库
    def save_to_database(self, title, content, sentiment, topic, db_session, News):
        """
        将新闻保存到数据库
        
        参数:
            title: 新闻标题
            content: 新闻内容
            sentiment: 情感分析结果
            topic: 主题分类结果
            db_session: 数据库会话
            News: 新闻模型类
        """
        try:
            # 创建新闻对象
            new_news = News(
                title=title,
                content=content,
                topic=topic,
                sentiment=sentiment
            )
            
            # 添加到数据库
            db_session.add(new_news)
            db_session.commit()
            
            print(f"新闻已成功保存到数据库，标题：{title}，主题：{topic}，情感：{sentiment}")
            return True
        except Exception as e:
            print(f"保存新闻到数据库时出错: {e}")
            db_session.rollback()
            return False

    #分析和保存新闻(使用上述两个函数)    
    def analyze_and_save_news(self, title, content, db_session, News):
        """
        分析新闻情感和主题，并保存到数据库
        
        参数:
            title: 新闻标题
            content: 新闻内容
            db_session: 数据库会话
            News: 新闻模型类
            
        返回:
            dict: 包含分析结果的字典
        """
        # 分析情感和主题
        sentiment, topic = self.analyze_news_sentiment_topic(content)
        

        # 创建文本嵌入向量
        text_for_embedding = f"{title} {content}"
        embedding = None
        embedding_pickle = None
        try:
            # 确保缓存目录存在
            cache_dir = "cache_data"
            os.makedirs(cache_dir, exist_ok=True)
            
            # 提取关键词
            keywords = batch_get_keywords([text_for_embedding])[0]
            keyword_text = "".join(keywords)
            
            # 生成嵌入向量
            embedding = batch_text_embedding([keyword_text])[0]
            
            embedding_list = embedding.tolist()
            embedding_pickle = pickle.dumps(embedding_list)
        except Exception as e:
            print(f"计算新闻嵌入向量时出错: {e}")
            embedding_pickle = None
        
        try:
            # 创建新闻对象
            new_news = News(
                title=title,
                content=content,
                topic=topic,
                sentiment=sentiment,
                embedding=embedding_pickle,
                upload_time=datetime.now(timezone(timedelta(hours=8)))
            )
            
            # 添加到数据库
            db_session.add(new_news)
            db_session.commit()
            
            # 将向量保存到 Elasticsearch
            if embedding is not None:
                try:
                    # 准备元数据
                    metadata = {
                        'news_id': new_news.id,
                        'title': title,
                        'content': content[:50],  # 截取部分内容作为预览
                        'topic': topic,
                        'sentiment': sentiment,
                        'upload_time': datetime.now(timezone(timedelta(hours=8)))
                    }
                    
                    # 索引到 Elasticsearch
                    index_news_vector(f"news_{new_news.id}", embedding, metadata)
                    print(f"新闻 {new_news.id} 已成功索引到 Elasticsearch")
                except Exception as e:
                    print(f"索引新闻到 Elasticsearch 失败: {str(e)}")
            
            print(f"新闻已成功保存到数据库，标题：{title}，主题：{topic}，情感：{sentiment}")
            return {
                "id": new_news.id,
                "title": title,
                "content": content[:200] + "..." if len(content) > 200 else content,
                "sentiment": sentiment,
                "topic": topic,
                "save_success": True
            }
        except Exception as e:
            print(f"保存新闻到数据库时出错: {e}")
            db_session.rollback()
            return {
                "title": title,
                "content": content[:200] + "..." if len(content) > 200 else content,
                "sentiment": sentiment,
                "topic": topic,
                "save_success": False,
                "error": str(e)
            }
        
    #借助大模型分析新闻要素  
    def analyze_news(self, title, content):
        """
        分析新闻内容，提取关键信息
        
        参数:
            title: 新闻标题
            content: 新闻正文
            
        返回:
            dict: 包含各类新闻要素的字典
        """
        # 构建系统提示词
        system_prompt = """
        作为新闻视觉分析师，需从新闻内容中提取以下7大要素，以JSON格式输出,每一个要素对应列表：
        要素规范（必填）：
        1. 核心事件（名词短语，一定要和标题紧密相关，3-5个）
        2. 关键动作（动宾结构，2-3个）
        3. 显著实体（具体名称，具体的人名和物品名和机构名一定要直接提取出来，不需要在后面注明是人名还是机构名）
        4. 场景特征（环境/时间/视觉元素）
        5. 情感内容(丰富)
        6. 视觉隐喻（象征性元素）
        7. 数据统计（新闻中的数据和统计信息）
        输出必须使用如下格式：
        {
            "核心事件": [...],
            "关键动作": [...],
            "显著实体": [...],
            "场景特征": [...],
            "情感内容": [...],
            "视觉隐喻": [...],
            "数据统计": [...]
        }
        """
        
        # 构建输入文本
        input_text = system_prompt + f"\n标题：{title}\n正文：{content}"
        
        # 调用LLM获取分析结果
        result = self.llm_client.ask(input_text)
        
        # 解析结果
        elements = self._parse_elements(result["answer"])
        
        print(elements)

        # 保存到Excel（可选）
        #self.save_to_excel(title, content, elements)
        
        return elements
    
    #新闻要素解析
    def _parse_elements(self, text):
        """解析LLM返回的JSON文本"""
        elements = {
            "核心事件": [], "关键动作": [], "显著实体": [],
            "场景特征": [], "情感内容": [], "视觉隐喻": [], "数据统计": []
        }
        try:
            json_block = text.split("```json")[-1].split("```")[0].strip()
            parsed_json = json.loads(json_block)
            # 转换繁体到简体
            converted_parsed = {cc.convert(k): v for k, v in parsed_json.items()}
            return {k: converted_parsed.get(cc.convert(k), []) for k in elements}
        except:
            return elements
    
    #保存新闻要素到Excel(不用)
    def save_to_excel(self, title, content, elements, excel_path="news_analysis.xlsx"):
        """将分析结果保存到Excel"""
        excel_data = {
            "标题": [title], 
            "正文": [content],
            "事件内容": [elements.get("核心事件", [])],
            "动作内容": [elements.get("关键动作", [])],
            "实体内容": [elements.get("显著实体", [])],
            "场景内容": [elements.get("场景特征", [])],
            "情感内容": [elements.get("情感内容", [])],
            "隐喻内容": [elements.get("视觉隐喻", [])],
            "数据内容": [elements.get("数据统计", [])]
        }
        df = pd.DataFrame(excel_data)
        df.to_excel(excel_path, index=False, engine='openpyxl')
        return excel_path

class ImageRecommender:
    """图片推荐器类"""
    
    def __init__(self, image_dir="./CNA_images", db_config=None):
        """
        初始化图片推荐器
        
        参数:
            image_dir: 图片目录路径
            db_config: 数据库配置字典，需包含host, user, password, database
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
        self.image_dir = os.path.join(backend_dir, image_dir)
        
        # 默认数据库配置
        self.db_config = db_config or {
            'host': 'localhost',
            'user': 'root',
            'password': '123456',
            'database': 'pixtock'
        }
        
        # 初始化数据库连接
        self._init_database()
        
        # 初始化CLIP模型 - 使用工具模块的懒加载函数
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "ViT-B-16"
        self.clip_model, self.clip_preprocess = load_cn_clip_model()
        if self.clip_model is None:
            print("警告：CLIP模型加载失败，图片推荐功能将无法正常工作")
        
        # 初始化缓存
        self.text_feature_cache = {}
        self.image_urls = {}  # 存储图片URL映射
        
        # 检查Elasticsearch是否可用
        try:
            from ..utils.es_utils import es_client
            self.es_available = es_client and es_client.ping()
            if self.es_available:
                print("Elasticsearch连接成功，将使用向量数据库进行检索")
            else:
                print("Elasticsearch不可用，将使用传统方法检索")
                # 在初始化时预处理图片并加载特征
                self.image_features = self.load_features_from_db()
                # 尝试从 Elasticsearch 加载新闻向量
                self.news_features = self.load_news_features_from_es()
        except Exception as e:
            print(f"检查Elasticsearch连接时出错: {e}")
            self.es_available = False
            # 在初始化时预处理图片并加载特征
            self.image_features = self.load_features_from_db()
            # 尝试从 Elasticsearch 加载新闻向量
            self.news_features = self.load_news_features_from_es()
    
    def _init_database(self):
        """初始化数据库连接和表"""
        try:
            self.conn = pymysql.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            print("数据库连接成功")
        except pymysql.Error as e:
            print(f"数据库连接或初始化错误: {e}")
            raise
    
    def _reconnect_if_needed(self):
        """如果连接断开，尝试重新连接数据库"""
        try:
            self.conn.ping(reconnect=True)
        except pymysql.Error as e:
            print(f"重新连接数据库时出错: {e}")
            self._init_database()
    
    #从数据库加载所有图片特征
    def load_features_from_db(self):
        """从数据库加载所有图片特征"""
        image_features = {}
        image_urls = {}  # 新增：存储图片URL
        
        # 如果使用Elasticsearch进行向量检索，则不需要预加载所有向量
        if hasattr(self, 'es_available') and self.es_available:
            print("使用Elasticsearch进行向量检索，无需预加载图片向量")
            return image_features
        
        self._reconnect_if_needed()
        
        try:
            # 从images表加载图片嵌入
            self.cursor.execute("""
                SELECT id, file_path, embedding, url 
                FROM images 
                WHERE embedding IS NOT NULL
            """)
            results = self.cursor.fetchall()
            
            if not results:
                print("数据库images表中没有找到图片嵌入，将尝试使用图片自动计算")
                return self.preprocess_images()
                
            print(f"从数据库加载 {len(results)} 条图片嵌入...")
            for row in tqdm(results, desc="加载嵌入"):
                try:
                    image_id, file_path, embedding_data, url = row
                    
                    # 获取文件名
                    if file_path and os.path.isfile(file_path):
                        image_name = os.path.basename(file_path)
                    else:
                        # 如果文件路径无效，使用ID作为标识
                        image_name = f"{image_id}.jpg"
                    
                    # 解析嵌入向量
                    vector = pickle.loads(embedding_data)
                    
                    # 确保向量已归一化
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = vector / norm
                    
                    # 存储到字典
                    image_features[image_name] = vector
                    image_urls[image_name] = url  # 存储图片URL
                    
                except Exception as e:
                    print(f"加载图片 {row[0]} 的嵌入时出错: {e}")
                    continue
                    
            print(f"成功加载 {len(image_features)} 条图片嵌入")
            # 保存图片URL映射
            self.image_urls = image_urls
            return image_features
            
        except pymysql.Error as e:
            print(f"从数据库加载数据时出错: {e}")
            self.image_urls = {}
            return {}
    
    def preprocess_images(self):
        """处理CNA_images文件夹中的图片，计算嵌入向量"""
        image_features = {}
        
        # 获取图片文件列表
        if not os.path.exists(self.image_dir):
            print(f"图片目录 {self.image_dir} 不存在")
            return {}
            
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        if not image_files:
            print(f"图片目录 {self.image_dir} 中没有找到图片")
            return {}
        
        # 确保CLIP模型已加载
        if self.clip_model is None:
            self.clip_model, self.clip_preprocess = load_cn_clip_model()
            if self.clip_model is None:
                print("错误：无法加载CLIP模型，无法处理图片")
                return {}
        
        print(f"处理 {len(image_files)} 张图片...")
        for img_name in tqdm(image_files, desc="计算图片嵌入"):
            try:
                # 加载并处理图片
                image_path = os.path.join(self.image_dir, img_name)
                image = Image.open(image_path).convert("RGB")
                inputs = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                
                # 计算嵌入向量
                with torch.no_grad():
                    feat = self.clip_model.encode_image(inputs).cpu().numpy()
                
                # 归一化特征
                normalized_feat = feat / np.linalg.norm(feat)
                
                # 存储到字典
                image_features[img_name] = normalized_feat
                
            except Exception as e:
                print(f"处理图片 {img_name} 时出错: {e}")
                continue
                
        print(f"成功计算 {len(image_features)} 张图片的嵌入向量")
        return image_features
    
    #编码文本特征
    def encode_text(self, text):
        """编码文本特征"""
        if not text.strip():
            return np.zeros((1, 512))
        
        # 确保CLIP模型已加载
        if self.clip_model is None:
            self.clip_model, self.clip_preprocess = load_cn_clip_model()
            if self.clip_model is None:
                print("错误：无法加载CLIP模型，无法编码文本")
                return np.zeros((1, 512))
            
        cache_key = hash(text)
        if cache_key in self.text_feature_cache:
            return self.text_feature_cache[cache_key]
            
        try:
            from cn_clip import clip
            tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                feat = self.clip_model.encode_text(tokens).cpu().numpy()
            
            normalized_feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
            self.text_feature_cache[cache_key] = normalized_feat
            return normalized_feat
        except Exception as e:
            print(f"编码文本特征时出错: {e}")
            return np.zeros((1, 512))
    
    #优化各要素权重
    def optimize_weights(self, news_elements, n_trials=200):
        """优化各要素权重"""
        # 使用全局固定权重设置，不再进行优化
        print("\n使用全局固定权重:")
        for field, weight in GLOBAL_WEIGHTS.items():
            print(f"{field}: {weight:.4f}")
        return GLOBAL_WEIGHTS.copy()
    
    def load_news_features_from_es(self):
        """从 Elasticsearch 加载新闻向量"""
        news_features = {}
        
        # 如果使用Elasticsearch进行向量检索，则不需要预加载所有向量
        if hasattr(self, 'es_available') and self.es_available:
            print("使用Elasticsearch进行向量检索，无需预加载新闻向量")
            return news_features
        
        try:
            # 尝试导入 Elasticsearch 工具
            try:
                from ..utils.es_utils import es_client, NEWS_INDEX_NAME
                if es_client is None:
                    print("Elasticsearch 客户端未初始化")
                    return {}
            except ImportError:
                print("无法导入 Elasticsearch 工具")
                return {}
            
            # 查询所有新闻向量
            query = {
                "query": {
                    "match_all": {}
                },
                "size": 1000  # 限制结果数量
            }
            
            response = es_client.search(
                index=NEWS_INDEX_NAME,
                body=query
            )
            
            hits = response.get("hits", {}).get("hits", [])
            print(f"从 Elasticsearch 加载 {len(hits)} 条新闻向量...")
            
            for hit in hits:
                try:
                    news_id = hit["_id"]
                    source = hit["_source"]
                    
                    # 获取向量
                    vector = np.array(source["vector"])
                    
                    # 确保向量已归一化
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = vector / norm
                    
                    # 存储到字典
                    news_features[news_id] = vector
                    
                    # 存储标题作为URL（用于显示）
                    title = source.get("title", "未知标题")
                    self.image_urls[news_id] = f"新闻: {title}"
                    
                except Exception as e:
                    print(f"处理新闻向量 {hit.get('_id')} 时出错: {e}")
                    continue
            
            print(f"成功加载 {len(news_features)} 条新闻向量")
            return news_features
            
        except Exception as e:
            print(f"从 Elasticsearch 加载新闻向量时出错: {e}")
            return {}
    
    #为新闻推荐图片
    def recommend_images(self, news_elements, title, content, top_n=10, use_fixed_weights=True, word_threshold=0):
        """
        为新闻推荐图片
        
        参数:
            news_elements: 新闻要素字典
            title: 新闻标题
            content: 新闻正文
            top_n: 返回的推荐图片数量
            use_fixed_weights: 是否使用固定权重，默认为True
            word_threshold: 正文长度阈值，决定采用哪种匹配策略
            
        返回:
            list: 包含(图片URL, 相似度)元组的列表
        """
        # 获取权重
        if use_fixed_weights:
            best_weights = GLOBAL_WEIGHTS.copy()
            print("\n使用全局固定权重:")
        else:
            # 使用优化权重（原始方法）
            best_weights = self.optimize_weights(news_elements)
            print("\n使用优化权重:")
            
        for field, weight in best_weights.items():
            print(f"{field}: {weight:.4f}")
        
        # 根据内容长度决定使用的策略
        content_length = len(content)
        
        # 使用Elasticsearch进行向量检索
        if hasattr(self, 'es_available') and self.es_available:
            try:
                from ..utils.es_utils import search_by_text_embedding, IMAGE_INDEX_NAME, NEWS_INDEX_NAME
                
                if content_length < word_threshold:
                    # 正文长度小于阈值，直接用标题+正文的文本嵌入计算相似度
                    combined_text = title + " " + content
                    print(f'正在匹配 {combined_text[:50]}...')  # 截断显示避免过长
                    text_embedding = self.encode_text(combined_text)
                    # 使用Elasticsearch搜索
                    results = search_by_text_embedding(text_embedding[0], top_k=top_n)
                else:
                    # 分别计算每个要素的相似度得分并加权
                    all_results = []
                    total_weight = sum(best_weights.values())
                    
                    if total_weight == 0:
                        print("权重之和为零，无法计算相似度")
                        return []
                    
                    # 对每个字段分别进行搜索，然后加权合并结果
                    for field, weight in best_weights.items():
                        text = ""
                        if field == "标题":
                            text = title
                        elif field in news_elements:
                            text = " ".join(news_elements.get(field, []))
                            
                        if text and weight > 0:  # 确保文本不为空且权重大于0
                            try:
                                text_embedding = self.encode_text(text)
                                # 为每个字段搜索更多结果，以便后续加权
                                field_results = search_by_text_embedding(
                                    text_embedding[0], 
                                    top_k=min(top_n * 3, 100),  # 获取更多候选结果
                                    index_name=IMAGE_INDEX_NAME
                                )
                                # 为每个结果添加权重
                                for item in field_results:
                                    item['weighted_score'] = item['score'] * (weight / total_weight)
                                all_results.extend(field_results)
                            except Exception as e:
                                print(f"处理特征 {field} 时出错: {e}")
                                continue
                    
                    # 如果没有任何结果，尝试使用标题
                    if not all_results:
                        print("加权特征为零向量，使用标题特征作为备选")
                        if title:
                            text_embedding = self.encode_text(title)
                            results = search_by_text_embedding(text_embedding[0], top_k=top_n, index_name=IMAGE_INDEX_NAME)
                        else:
                            print("无有效文本特征")
                            return []
                    else:
                        # 合并结果：按ID分组，累加加权分数
                        results_dict = {}
                        for item in all_results:
                            item_id = item['id']
                            if item_id in results_dict:
                                results_dict[item_id]['weighted_score'] += item.get('weighted_score', 0)
                            else:
                                results_dict[item_id] = item
                                if 'weighted_score' not in item:
                                    item['weighted_score'] = 0
                        
                        # 按加权分数排序
                        results = sorted(
                            results_dict.values(), 
                            key=lambda x: x.get('weighted_score', 0), 
                            reverse=True
                        )[:top_n]
                
                # 构建结果列表
                recommendations = []
                for item in results:
                    item_id = item['id']
                    score = item.get('weighted_score', item['score'] - 1.0)  # 减去1.0是因为ES的cosine相似度计算加了1.0
                    
                    # 获取URL
                    item_url = item.get('url', '')
                    if not item_url:
                        # 如果没有URL，使用默认路径构建
                        item_url = f"http://localhost:5000/dataimages2/{item_id}"
                    
                    recommendations.append((item_url, float(score)))
                
                return recommendations
                
            except Exception as e:
                print(f"使用Elasticsearch检索时出错: {e}")
                print("将回退到传统方法")
        
        # 传统方法：使用预加载的向量进行内存中计算
        # 检查图片特征是否为空
        if not hasattr(self, 'image_features'):
            self.image_features = self.load_features_from_db()
        
        if not hasattr(self, 'news_features'):
            self.news_features = self.load_news_features_from_es()
        
        if not self.image_features and not self.news_features:
            print("没有找到任何有效图片或新闻向量")
            return []
        
        # 合并图片和新闻特征
        all_features = {}
        all_features.update(self.image_features)
        all_features.update(self.news_features)
        
        # 准备图片特征矩阵以便后续计算相似度
        item_names = list(all_features.keys())
        item_features = np.array([all_features[name] for name in item_names])
        
        # 根据内容长度决定使用的策略
        if content_length < word_threshold:
            # 正文长度小于阈值，直接用标题+正文的文本嵌入计算相似度
            combined_text = title + " " + content
            print(f'正在匹配 {combined_text[:50]}...')  # 截断显示避免过长
            text_embedding = self.encode_text(combined_text)
            similarities = cosine_similarity(text_embedding, item_features)[0]
        else:
            # 分别计算每个要素的相似度得分并加权
            weighted_scores = np.zeros(len(item_features))
            total_weight = sum(best_weights.values())
            
            if total_weight == 0:
                print("权重之和为零，无法计算相似度")
                return []
            
            for field, weight in best_weights.items():
                text = ""
                if field == "标题":
                    text = title
                elif field in news_elements:
                    text = " ".join(news_elements.get(field, []))
                    
                if text and weight > 0:  # 确保文本不为空且权重大于0
                    try:
                        text_embedding = self.encode_text(text)
                        field_similarities = cosine_similarity(text_embedding, item_features)[0]
                        weighted_scores += (weight / total_weight) * field_similarities
                    except Exception as e:
                        print(f"处理特征 {field} 时出错: {e}")
                        continue
            
            # 如果所有特征都处理失败，尝试使用标题
            if np.sum(weighted_scores) == 0:
                print("加权特征为零向量，使用标题特征作为备选")
                if title:
                    text_embedding = self.encode_text(title)
                    similarities = cosine_similarity(text_embedding, item_features)[0]
                else:
                    print("无有效文本特征")
                    return []
            else:
                similarities = weighted_scores
        
        # 获取排序后的索引
        top_indices = np.argsort(-similarities)[:top_n]
        
        # 构建结果列表
        recommendations = []
        for idx in top_indices:
            item_name = item_names[idx]
            sim_value = float(similarities[idx])
            
            # 使用URL而不是文件名
            item_url = self.image_urls.get(item_name, "")
            if not item_url:
                # 如果没有URL，使用默认路径构建
                item_url = f"http://localhost:5000/dataimages2/{item_name}"
            
            recommendations.append((item_url, sim_value))
        
        return recommendations

    def __del__(self):
        """析构函数，确保关闭数据库连接"""
        try:
            if hasattr(self, 'cursor') and self.cursor:
                self.cursor.close()
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except Exception as e:
            print(f"关闭数据库连接时出错: {e}")

# 以下是权重管理相关函数

def get_current_weights():
    """获取当前全局权重配置"""
    return GLOBAL_WEIGHTS.copy()

def set_weights(new_weights, is_admin=False):
    """
    设置全局权重配置
    
    参数:
        new_weights: 新的权重字典
        is_admin: 是否是管理员，只有管理员才能修改权重
        
    返回:
        bool: 是否成功更新权重
    """
    if not is_admin:
        print("只有管理员才能修改权重配置")
        return False
    
    return update_weights(new_weights)

def main():
    """主函数示例"""
    # 初始化时间
    start_time = time.time()
    
    # 数据库配置 - 根据实际情况修改
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '123456',
        'database': 'pixtock'
    }
    
    # 示例新闻
    title = "湖北学霸杨元元带母上学，最终厕所自缢身亡，活生生被母亲吸干血"
    content = """
声明：本文陈述内容参考的"官方信息来源"，均赘述在文章末尾，感谢支持。文|烟寒初妆编辑 |烟寒初妆"没人愿意一辈子被脐带拴住，我终于可以做回自己了。"2009年11月26日，上海海事大学女研究生杨元元在留下这句遗言后，便在宿舍厕所结束了自己年仅三十岁的生命。事发后，女生的母亲和弟弟集结了一大批亲戚，跑到学校闹事，逼迫校方赔他们35万元，完全看不出对自己女儿逝世的痛惜。这一闹吸引来无数媒体记者，此事件成了当时的热门。人们纷纷猜测：大好的年华，她究竟为什么会如此毅然决然赴死？是否和这对闹事的母子有关？被母爱"绑架"的童年杨元元，1979年出生在湖北宜昌市的一个双职工家庭。父亲是那个年代罕见的大学生，在化工厂当技术员，收入不菲，母亲因为读初中时做了知青，没有继续求学，所以只是普通工人。杨元元下面还有一个弟弟，由于父亲收入不错，他们一家四口在当时过得还算可以。不幸的是，在杨元元六岁那年，父亲因病去世了。杨元元的母亲望瑞玲没有多少文化，又只是个普通工人，丈夫一死，养家糊口和照顾两个孩子的重担全都落在了她的肩上。靠着微薄的收入，望瑞玲养活孩子养得非常艰难，孤儿寡母在弱肉强食的社会也很容易被人欺凌。望瑞玲觉得自己是吃了没文化的亏，所以经常告诫两个孩子要好好读书，杨元元和弟弟也很争气，从小到大一直是品学兼优的"别人家孩子"，从不让她操一点心。尽管儿女都很懂事和优秀，生活的负担还是压得望瑞玲喘不过气。在这种情况下，她原本就有些暴躁的性格变得越来越极端，动不动就发牢骚。可能因为杨元元是女孩，又是家里的长女，望瑞玲完全把她当成了出气筒，每天都对着她埋怨个不停："要不是因为你，我哪用得着这么累，都是因为你。"望瑞玲的怨气全都被女儿消化吸收了，可她根本没有想过，幼小的女儿又是靠什么来宣泄这些怨气呢？杨元元知道母亲辛苦，虽然这并不是她的错：养育子女本就是父母的责任，但她却无法反驳，因为那必然会招来母亲更凶狠的斥骂。所以，多年以来，她都习惯了默默的忍耐，母亲让她干什么她就干什么。很多时候，杨元元看着窗外的天空，期望着：只要我努力学习，考上好大学，再找到工作挣钱养自己，就能逃离这里了。可惜，她一辈子都没有逃出母亲的魔爪。活在母亲的监视下1998年，杨元元凭借优异的高考成绩，进入武汉大学经济系。报考这个专业，是因为她觉得学经济出来以后可以从事金融等相关专业，能挣大钱，只要自己有钱了，就再也不用过以前的苦日子了。因为家里经济条件太差，望瑞玲拿不出钱给女儿交学费，杨元元在大学期间只能凭助学贷款和勤工俭学维持。尽管如此，不再受母亲管束的她日子快活极了，对她来说，就算再苦再累，也比天天被母亲折磨要好得多。这种自由自在的生活才过了两年，一切戛然而止。2000年，杨元元的弟弟也考上了武汉大学，双喜临门，亲戚朋友纷纷来找望瑞玲道喜，可是望瑞玲却愁眉不展。原来，望瑞玲一直住在工厂宿舍，可是这一年工厂搬迁新地址，没有宿舍提供给她。如果员工跟过去工作，还得自己买房，可能要好几万。望瑞玲哪里掏得出这么多钱，不满五十岁的她直接办理提前退休。退休后该去哪呢？按理说，杨家在老家还有一座老房子，望瑞玲可以去那里安居。但她却想：我给武汉大学培养了两个大学生，为啥不上武汉大学住？这种想法在任何人听来都匪夷所思，望瑞玲却毫不自知，她很得意自己能想出这么好的办法，当即收拾行李赶往武汉大学，完全不提前通知女儿，就不由分说地搬进了女儿的宿舍。大学宿舍本来就小，她一进来空间更逼仄了，她还在里面做饭，弄得整个屋子乌烟瘴气的。杨元元试图劝说母亲离开，可每次她刚一开口，望瑞玲滔滔不绝的唠叨和责骂就扑面而来："我是不是白养你啦？你连我的话都不听了？当初要不是我养你，你早就饿死在路边了……"唾沫星子喷她一脸。没有办法，杨元元只能选择闭嘴。杨元元治不了望瑞玲，她的室友可不吃这一套，她们跟望瑞玲非亲非故，凭什么忍着她在宿舍里胡乱折腾。几个女孩当即跟望瑞玲大吵一架，可是十几岁的女孩哪里吵得过五十岁的泼妇，没办法，她们只能找到校领导，要求把望瑞玲赶出去。别说几个小姑娘了，望瑞玲撒泼打滚那一套，连校领导都拿她没办法，最后迫于无奈，学校只能给杨元元找了个单独的宿舍，供她们母女俩住。望瑞玲每天都监视杨元元，连刷牙洗脸这种小事都不放过，还天天念叨："我一个人把你养大多不容易啊，你以后什么事都要听我的。"杨元元在这种生活下根本没有任何隐私和尊严，同学们也都对她避之不及，没有倾诉对象的杨元元每天都生活在痛苦之中。但这个时候她仍然怀揣着希望：等我毕业找到工作了，就可以自由了吧？然而，她远远低估了母亲的控制欲。剪不断的脐带女儿毕业后，望瑞玲还是不肯放过她，不停地掺和女儿找工作的事。杨元元通过优秀的表现，先后考上两个地方的公务员，后来还去大学当任课老师，这些明明已经是社会上非常体面的工作了，望瑞玲却一点都不知足，天天骂她："没出息！你待在这种小地方能混出什么样？我一直跟你说让你往高处走，去北京上海，你就是不听，我辛辛苦苦养你到底有什么用？"其实，发达城市人才济济，竞争激烈，并且生活成本极高，与其去那里，还不如在小地方拿一份稳当的工资悠闲地过日子。何况就凭杨元元的学历，她毕业后的工资在小城市也算中上水平了。望瑞玲可不管这些，只顾由着自己性子来。眼见好好的工作机会都被母亲搅黄，杨元元只好先去打短工，否则生存都是个问题了。她干过英语培训，还卖过保险，赚来的钱都被母亲拿去花销和供弟弟读书。再后来，弟弟考上了北大的博士，望瑞玲一点都不体恤女儿工作的辛苦，还挖苦她说："你看看你弟弟，再看看你，都是我生的，你怎么就那么没用……"她仿佛是忘记了，如果不是杨元元在这里辛辛苦苦打工，她宝贝儿子读博士的费用从哪里来？杨元元实在不甘心日子这样过下去，年近三十岁的时候，她决定考研，再次用知识重改自己的命运。一听女儿要考研，望瑞玲也大力支持，但她支持的理由并不是希望女儿越过越好，而是想让女儿带她过上好日子。她强硬要求杨元元考上海的学校，理由就是上海是个发达的大城市，根本不问杨元元自己的想法。而早就习惯于母亲管制的杨元元也是，母亲说什么她就听什么。通过一段时间的努力学习，杨元元如愿被上海海事大学法律专业录取。没想到的是，杨元元前脚刚到学校报到，后脚望瑞玲就跟来了。她简直像个巨婴，女儿走哪她跟哪，非要跟女儿一起住宿舍。这一次学校是坚决反对，不管望瑞玲怎么躺地上撒泼都没用。在望瑞玲还逼迫女儿给学校写申请书，可是女儿惯着她，学校可不惯着她，说不让住就是不让。最后还是一个老师可怜杨元元，帮她们在校外找了个每月450元的小房子。在上海，450元能租什么房子可想而知：空间极其窄小，只有两三平米，并且没做任何装修，连一张床都没有，母女俩只能大冬天睡地板。如果一开始望瑞玲回老家住，杨元元住学校宿舍，也不至于这么艰难。不止如此，杨元元还得每天打零工赚取租金、学费以及她两人的生活费，平时还得上课，这样的压力逼得她快要窒息。后来她终于受不了了，跟母亲谎称要回学校争取一下宿舍，结果两天过去却一直没回来。望瑞玲还以为女儿在躲她，跑到学校里闹。学校也不知道杨元元的踪迹，大家这才发现她失踪了。很快，学校在宿舍里发现了自缢身亡的杨元元。更令人惊骇的是：她是用两条系在一起的毛巾，将身体悬挂在卫生间的水龙头上，半蹲着勒死自己的！这个一辈子都像个提线木偶的女孩，唯一亲手做过决定的事就是自己的死亡。望瑞玲得知这个消息后，没有任何的悲痛和自责，而是借此机会朝学校索要赔款："你们害死了我女儿，我怎么办，你们快点赔我35万让我买房子住！"学校认为错不在自己，但此事毕竟闹得太大，为了息事宁人，学校只好赔了她16万。结语虽然社会倡导孝道，但不代表我们要愚孝。父母抚养儿女，儿女赡养父母是相对的，双方都不该对彼此有过分的要求，像望瑞玲这样的人，就是完全没有把女儿当成独立的人，而是当成自己随意使唤的工具，勒死杨元元的，并不是那两条毛巾，而是母亲的脐带。我们衷心希望，现代社会不要再出现这样的悲剧。
"""  # 此处是新闻正文
    
    # 步骤1：提取新闻信息
    analyzer = NewsAnalyzer()
    news_elements = analyzer.analyze_news(title, content)
    
    # 步骤2：推荐图片
    image_dir = "./CNA_images"
    recommender = ImageRecommender(image_dir, db_config)
    
    # 使用固定权重
    recommendations = recommender.recommend_images(news_elements, title, content, use_fixed_weights=True)
    
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
    result={
        "news_id":"",
        "news_title":"",
        "features":features,
        "recommended_images":[{
            "image_path":img_url,
            "similarity_score":float(score)
        }
        for img_url, score in recommendations
        ]
    }

    print(result)
