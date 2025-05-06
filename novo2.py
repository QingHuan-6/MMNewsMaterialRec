# -*- coding: utf-8 -*-
import csv
import pandas as pd
import torch
import jieba
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import os
import hashlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from LLM import LLMClient
import json

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # 避免内存碎片化
os.environ['TOKENIZERS_PARALLELISM'] = 'true'  # 启用并行化

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 配置参数 ==================
CACHE_DIR = "cache_data"  # 统一缓存目录
HISTORY_EMB_CACHE = os.path.join(CACHE_DIR, "history_embeddings.pkl")  # 历史文章向量缓存
KEYWORD_CACHE = os.path.join(CACHE_DIR, "keywords_cache.pkl")  # 关键词缓存
BATCH_SIZE = 32  # 批量处理大小
NUM_WORKERS = 4  # 并行工作线程数
WEIGHTS_CONFIG_PATH = "article_weights_config.json"  # 文章推荐权重配置文件

# 默认权重配置
DEFAULT_ARTICLE_WEIGHTS = {
    "content_similarity": 0.4,  # 内容相似度权重
    "theme_match": 0.3,         # 主题匹配权重
    "sentiment_match": 0.3      # 情感匹配权重
}

# 全局权重变量
ARTICLE_WEIGHTS = DEFAULT_ARTICLE_WEIGHTS.copy()

# 加载权重配置
def load_article_weights():
    """加载文章推荐权重配置"""
    global ARTICLE_WEIGHTS
    try:
        if os.path.exists(WEIGHTS_CONFIG_PATH):
            with open(WEIGHTS_CONFIG_PATH, 'r', encoding='utf-8') as f:
                weights = json.load(f)
                # 验证权重配置是否完整
                if all(k in weights for k in DEFAULT_ARTICLE_WEIGHTS.keys()):
                    # 确保权重和为1
                    total = sum(weights.values())
                    if abs(total - 1.0) > 0.01:  # 允许0.01的误差
                        weights = {k: v / total for k, v in weights.items()}
                    ARTICLE_WEIGHTS = weights
                    print(f"已加载文章推荐自定义权重配置: {ARTICLE_WEIGHTS}")
                    return ARTICLE_WEIGHTS
    except Exception as e:
        print(f"加载文章推荐权重配置出错: {e}")
    
    print(f"使用默认文章推荐权重配置: {ARTICLE_WEIGHTS}")
    return ARTICLE_WEIGHTS

# 保存权重配置
def save_article_weights(weights):
    """保存文章推荐权重配置"""
    try:
        # 验证权重配置
        if not all(k in weights for k in DEFAULT_ARTICLE_WEIGHTS.keys()):
            print("权重配置不完整，保存失败")
            return False
        
        # 确保权重和为1
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:  # 允许0.01的误差
            weights = {k: v / total for k, v in weights.items()}
        
        with open(WEIGHTS_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(weights, f, ensure_ascii=False, indent=2)
        
        # 更新全局变量
        global ARTICLE_WEIGHTS
        ARTICLE_WEIGHTS = weights
        
        print(f"文章推荐权重配置已保存: {WEIGHTS_CONFIG_PATH}")
        return True
    except Exception as e:
        print(f"保存文章推荐权重配置出错: {e}")
        return False

# 获取当前权重
def get_article_weights():
    """获取当前文章推荐权重配置"""
    return ARTICLE_WEIGHTS

# 设置权重
def set_article_weights(new_weights, is_admin=False):
    """更新文章推荐权重配置（仅管理员可用）"""
    if not is_admin:
        print("只有管理员可以更新权重配置")
        return False
    
    return save_article_weights(new_weights)

# 初始化加载权重
load_article_weights()

# ================== 初始化模型 ==================
# 中文BERT模型（关键词提取）
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese').eval().to(device)

# Sentence-BERT模型（相似度计算）
ST_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
st_tokenizer = AutoTokenizer.from_pretrained(ST_MODEL_NAME)
st_model = AutoModel.from_pretrained(ST_MODEL_NAME).eval().to(device)

# 初始化LLM客户端
llm_client = LLMClient()

# ================== 情感和主题定义 ==================
# 定义情感分类
SENTIMENTS = ["负面", "中性", "正面"]

# 定义主题分类
TOPICS = ["体育", "娱乐", "科技", "时政", "财经", "社会", "国际", "军事", "教育", "生活", "时尚"]

# ================== 数据预处理优化 ==================
# 预加载停用词
STOPWORDS = {'的', '了', '在', '是', '和', '等', '为', '对', '并', '这', '就', '也', '都', '一个'}


def preprocess_text(text):
    """预处理文本的优化实现"""
    return re.sub(r'\s+', '', text).replace('\n', '').replace('\t', '')


def process_long_text(text, max_length=500):
    """处理长文本，分割成多个块"""
    if len(text) <= max_length:
        return [text]
    
    # 简单的分块策略：按照句号分割
    chunks = []
    sentences = re.split(r'[。！？]', text)
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_length:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


# ================== 带缓存的批处理关键词提取 ==================
def batch_process_texts(texts, batch_size=32):
    """批量处理文本的分块"""
    all_chunks = []
    for text in texts:
        text = preprocess_text(text)
        chunks = process_long_text(text)
        all_chunks.extend(chunks)
    return all_chunks


def batch_get_word_vectors(words):
    """批量获取词向量"""
    unique_words = list(set(words))
    chunk_size = 100  # 可根据实际情况调整分块大小
    word_vectors = {}
    for i in range(0, len(unique_words), chunk_size):
        chunk = unique_words[i:i + chunk_size]
        inputs = bert_tokenizer(chunk, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        for j, word in enumerate(chunk):
            word_vectors[word] = outputs.last_hidden_state[j].mean(dim=0).cpu().numpy()
        del inputs, outputs
        torch.cuda.empty_cache()
    return word_vectors


def batch_get_keywords(texts, top_k=15):
    """批量获取关键词（带缓存）"""
    # 尝试加载缓存
    if os.path.exists(KEYWORD_CACHE):
        cache = pd.read_pickle(KEYWORD_CACHE)
        if len(cache) == len(texts):
            return cache

    # 批量处理流程
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        chunks = list(executor.map(preprocess_text, texts))

    # 批量分词
    word_batches = []
    for text in chunks:
        words = [w for w in jieba.lcut(text) if w not in STOPWORDS and len(w) > 1]
        word_batches.append(words)

    # 批量获取词向量
    all_words = list(set([w for batch in word_batches for w in batch]))
    word_vectors = batch_get_word_vectors(all_words)

    # 批量计算文档向量
    doc_vectors = []
    for words in word_batches:
        vecs = [word_vectors[w] for w in words if w in word_vectors]
        doc_vectors.append(np.mean(vecs, axis=0) if vecs else np.zeros(768))

    # 计算关键词得分
    keyword_results = []
    for words, doc_vec in zip(word_batches, doc_vectors):
        word_scores = {}
        for w in set(words):
            if w in word_vectors:
                similarity = cosine_similarity([word_vectors[w]], [doc_vec])[0][0]
                word_scores[w] = similarity * words.count(w)
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        keyword_results.append([w[0] for w in sorted_words])

    # 保存缓存
    pd.to_pickle(keyword_results, KEYWORD_CACHE)
    return keyword_results


# ================== 优化的向量生成 ==================
def batch_text_embedding(texts):
    """批量生成文本向量"""
    # 生成缓存标识
    hash_str = hashlib.md5("|".join(texts).encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"batch_{hash_str}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path)

    # 批量编码
    inputs = st_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = st_model(**inputs)

    embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
    np.save(cache_path, embeddings)
    return embeddings


# ================== 使用LLM进行新闻分析 ==================
def analyze_news(news):
    """使用LLM分析新闻的情感和主题"""
    # 分析情感
    sentiment_prompt = f"""你是一个专业的新闻情感分析师。请根据新闻内容，分析新闻文章的情感（负面/中性/正面）。
    只输出情感类型，若新闻未提及任何主题相关内容。情感包括：{', '.join(SENTIMENTS)}。
    情感的判断如下：
    1. 正面：新闻内容若主要包含积极、肯定、赞扬的表达，则可判定为正面情感。例如："某公司成功研发出新型产品，市场反响良好。"
    2. 中性：新闻内容若只是客观陈述事实，不带有明显的情感偏向，即为中性情感。例如："今天的气温是25摄氏度。"
    3. 负面：若新闻内容包含消极、否定、批判的表达，则为负面情感。例如："某企业因违规操作被处罚。"
    请确保输出格式正确，无需其他解释。
    
    新闻内容为："{news}"
    请你分析新闻的情感。
    """
    
    # 调用LLM获取情感分析结果
    sentiment_result = llm_client.ask(sentiment_prompt)
    sentiment = sentiment_result["answer"].strip()
    
    # 后处理情感结果，确保得到有效输出
    if sentiment not in SENTIMENTS:
        # 增加更严格的后处理，若不匹配则尝试根据关键词判断
        for keyword in ["积极", "肯定", "赞扬"]:
            if keyword in news:
                sentiment = "正面"
                break
        else:
            for keyword in ["消极", "否定", "批判"]:
                if keyword in news:
                    sentiment = "负面"
                    break
            else:
                sentiment = "中性"
    
    # 分析主题
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
    
    新闻内容为："{news}"
    请你进行主题分类。
    """
    
    # 调用LLM获取主题分析结果
    topic_result = llm_client.ask(topic_prompt)
    topic = topic_result["answer"].strip()
    
    # 后处理主题结果，确保得到有效输出
    if topic not in TOPICS:
        topic = "社会"  # 默认主题
    
    return sentiment, topic


# ================== 历史文章预加载 ==================
def load_history_data(history_file):
    """加载并预处理历史数据"""
    # 检查文件扩展名
    if history_file.endswith('.xlsx') or history_file.endswith('.xls'):
        df = pd.read_excel(history_file)
        # 检查并重命名Excel文件的列名
        if '新闻内容' in df.columns and '主题' in df.columns and '分类结果' in df.columns:
            # 如果是Excel格式且列名已经是新格式，直接使用
            pass
        elif 'content' in df.columns and 'theme' in df.columns and 'label' in df.columns:
            # 如果是旧的列名，重命名为新格式
            df = df.rename(columns={'content': '新闻内容', 'theme': '主题', 'label': '分类结果'})
        else:
            # 如果列名不匹配，尝试使用默认列名
            print("警告：Excel格式列名不匹配，将使用默认列名")
            if len(df.columns) >= 3:
                df.columns = ['新闻内容', '主题', '分类结果'] + list(df.columns[3:])
    else:
        # 如果是CSV文件或其他格式，假设它是旧的格式
        df = pd.read_csv(history_file, header=None, names=['新闻内容', '主题', '分类结果'])
    
    # 去重
    df = df.drop_duplicates(subset=['新闻内容'])

    # 预生成所有历史文章的向量
    if not os.path.exists(HISTORY_EMB_CACHE):
        print("正在预计算历史文章向量...")
        all_keywords = batch_get_keywords(df['新闻内容'].tolist())
        keyword_texts = ["".join(kws) for kws in all_keywords]

        # 分批处理避免内存溢出
        embeddings = []
        for i in tqdm(range(0, len(keyword_texts), BATCH_SIZE)):
            batch = keyword_texts[i:i + BATCH_SIZE]
            embeddings.append(batch_text_embedding(batch))

        df['embedding'] = np.vstack(embeddings).tolist()
        pd.to_pickle(df, HISTORY_EMB_CACHE)
    else:
        df = pd.read_pickle(HISTORY_EMB_CACHE)
        # 确保加载的DataFrame有正确的列名
        if 'content' in df.columns and '新闻内容' not in df.columns:
            df = df.rename(columns={'content': '新闻内容'})
        if 'theme' in df.columns and '主题' not in df.columns:
            df = df.rename(columns={'theme': '主题'})
        if 'label' in df.columns and '分类结果' not in df.columns:
            df = df.rename(columns={'label': '分类结果'})
        if '新闻文本' in df.columns and '新闻内容' not in df.columns:
            df = df.rename(columns={'新闻文本': '新闻内容'})
        if '情感' in df.columns and '分类结果' not in df.columns:
            df = df.rename(columns={'情感': '分类结果'})

    # 验证列名是否正确
    required_columns = ['新闻内容', '主题', '分类结果', 'embedding']
    for col in required_columns:
        if col not in df.columns:
            print(f"错误: 缺少必要的列 '{col}'，当前列: {df.columns.tolist()}")
            # 如果是核心列缺失，尝试从类似列复制
            if col == '新闻内容' and 'content' in df.columns:
                df['新闻内容'] = df['content']
            elif col == '主题' and 'theme' in df.columns:
                df['主题'] = df['theme']
            elif col == '分类结果' and 'label' in df.columns:
                df['分类结果'] = df['label']
            # 如果没有可复制的列，创建默认值
            elif col == '新闻内容':
                df['新闻内容'] = ['无内容'] * len(df)
            elif col == '主题':
                df['主题'] = ['社会'] * len(df)
            elif col == '分类结果':
                df['分类结果'] = ['中性'] * len(df)
    
    # 打印列名以便调试
    print(f"最终处理后的数据集列名: {df.columns.tolist()}")
    return df


# ================== 从数据库中加载新闻数据的函数 ==================
def load_news_from_db(db_session, News, top_n=None):
    """从数据库加载新闻数据
    
    Args:
        db_session: 数据库会话
        News: News模型类
        top_n: 限制返回的记录数量
    
    Returns:
        包含新闻数据的DataFrame
    """
    import pickle
    import pandas as pd
    
    # 查询有嵌入向量的新闻
    query = db_session.query(News).filter(News.embedding != None)
    
    # 如果指定了数量限制
    if top_n:
        query = query.limit(top_n)
    
    news_records = query.all()
    
    if not news_records:
        return None
    
    # 构建DataFrame
    df = pd.DataFrame({
        'id': [news.id for news in news_records],
        'title': [news.title for news in news_records],
        '新闻内容': [news.content for news in news_records],
        '主题': [news.topic for news in news_records],
        '分类结果': [news.sentiment for news in news_records],
        'embedding': [pickle.loads(news.embedding) for news in news_records],
        'upload_time': [news.upload_time for news in news_records]
    })
    
    return df


# ================== 修改后的推荐函数 ==================
def recommend_articles(input_title, input_content, db_session=None, News=None, history_file="news_history.xlsx", top_n=5, use_db=False):
    """优化后的推荐函数，支持从Excel文件或数据库加载文章
    
    Args:
        input_title: 输入标题
        input_content: 输入内容
        db_session: 数据库会话（当use_db=True时使用）
        News: News模型类（当use_db=True时使用）
        history_file: 历史文章Excel文件路径（当use_db=False时使用）
        top_n: 返回推荐结果的数量
        use_db: 是否使用数据库加载文章
    
    Returns:
        推荐结果列表
    """
    # 确保缓存目录存在
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 合并输入文本
    full_text = f"{input_title}。{input_content}" if input_title else input_content
    
    # 使用LLM分析情感和主题
    sentiment_result, topic_result = analyze_news(full_text)
    print(f"输入文章情感分析结果: {sentiment_result}")
    print(f"输入文章主题分类结果: {topic_result}")
    
    # 加载历史数据
    if use_db and db_session and News:
        # 从数据库加载
        history_df = load_news_from_db(db_session, News)
        if history_df is None or len(history_df) == 0:
            print("数据库中没有找到有效的新闻记录，将使用历史文件")
            history_df = load_history_data(history_file)
    else:
        # 从Excel/CSV文件加载（带缓存）
        history_df = load_history_data(history_file)

    # 清除输入文章的关键词缓存
    if os.path.exists(KEYWORD_CACHE):
        cache = pd.read_pickle(KEYWORD_CACHE)
        if len(cache) > 0:
            cache = cache[:-1]
            pd.to_pickle(cache, KEYWORD_CACHE)

    # 处理输入文章
    input_keywords = batch_get_keywords([full_text])[0]

    # 清除输入文章的向量缓存
    hash_str = hashlib.md5("".join(input_keywords).encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"batch_{hash_str}.npy")
    if os.path.exists(cache_path):
        os.remove(cache_path)

    input_emb = batch_text_embedding(["".join(input_keywords)])[0]

    # 批量计算相似度
    history_embs = np.array(history_df['embedding'].tolist())
    similarities = cosine_similarity([input_emb], history_embs)[0]

    # 确保列名存在
    if '主题' not in history_df.columns:
        print(f"警告: '主题'列不存在，可用列: {history_df.columns.tolist()}")
        # 尝试查找可能的替代列
        if 'theme' in history_df.columns:
            history_df = history_df.rename(columns={'theme': '主题'})
        else:
            # 如果没有替代列，添加一个全为默认值的列
            history_df['主题'] = ['社会'] * len(history_df)
            
    if '分类结果' not in history_df.columns:
        print(f"警告: '分类结果'列不存在，可用列: {history_df.columns.tolist()}")
        # 尝试查找可能的替代列
        if 'label' in history_df.columns:
            history_df = history_df.rename(columns={'label': '分类结果'})
        elif '情感' in history_df.columns:
            history_df = history_df.rename(columns={'情感': '分类结果'})
        else:
            # 如果没有替代列，添加一个全为默认值的列
            history_df['分类结果'] = ['中性'] * len(history_df)

    # 主题匹配得分
    theme_scores = [1 if history_df.iloc[i]['主题'] == topic_result else 0 for i in range(len(history_df))]

    # 情感匹配得分
    sentiment_scores = [1 if history_df.iloc[i]['分类结果'] == sentiment_result else 0 for i in range(len(history_df))]

    # 综合得分 - 使用可配置的权重
    # 获取当前权重配置
    weights = get_article_weights()
    content_weight = weights["content_similarity"]
    theme_weight = weights["theme_match"]
    sentiment_weight = weights["sentiment_match"]

    # 应用权重计算综合得分
    combined_scores = [
        content_weight * s + theme_weight * t + sentiment_weight * k 
        for s, t, k in zip(similarities, theme_scores, sentiment_scores)
    ]

    # 获取TopN结果
    top_indices = np.array(combined_scores).argsort()[-top_n:][::-1]
    results = []
    for idx in top_indices:
        row = history_df.iloc[idx]
        
        # 构建结果对象（数据库模式和CSV模式字段略有不同）
        if use_db and 'id' in row:
            result = {
                "id": int(row['id']),
                "title": row['title'],
                "content": row['新闻内容'] if '新闻内容' in row else row.get('content', ''),
                "topic": row['主题'],
                "sentiment": row['分类结果'],
                "similarity": round(float(combined_scores[idx]), 4)
            }
        else:
            result = {
                "id": idx,
                "content": row['新闻内容'] if '新闻内容' in row else row.get('content', ''),
                "theme": row['主题'],
                "label": row['分类结果'],
                "similarity": round(float(combined_scores[idx]), 4)
            }
            
        results.append(result)

    return results


# ================== 使用示例 ==================
if __name__ == "__main__":
    # 初始化缓存目录
    os.makedirs(CACHE_DIR, exist_ok=True)

    sample_title = ""
    sample_content = "臨近國慶節，香港中環街頭掛滿了慶祝中華人民共和國成立75周年標語，營造濃厚的慶祝國慶氛圍。（香港中通社記者  謝光磊 攝）  香港中通社圖片"


    recommendations = recommend_articles(sample_title, sample_content)

    print(f"输入文章: {sample_title}\n{sample_content[:100]}...")
    print("\n推荐历史文章TOP5:")
    for i, item in enumerate(recommendations, 1):
        print(f"{i}. [相似度{item['similarity']:.4f}] {item['content'][:50]}...")
