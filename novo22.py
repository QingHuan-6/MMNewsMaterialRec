from concurrent.futures import ThreadPoolExecutor
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # 避免内存碎片化
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import json
from tqdm import tqdm
import csv
import jieba
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os
import hashlib
import pickle
from LLM import LLMClient







# 初始化LLM客户端
llm_client = LLMClient()

# 明确指定设备为第二块 GPU（cuda:1）
device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")

# 不再需要加载Meta-Llama模型和分词器
# model_id = "/home/data_SSD/zx/weight/Meta-Llama-3-8B-Instruct-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map={"": device}  # 指定模型加载到指定设备
# )

# 定义情感分类和系统提示
SENTIMENTS = [
    "负面", "中性", "正面"
]

# 增加示例和更详细的判断标准
SENTIMENT_PROMPT = """你是一个专业的新闻情感分析师。请根据新闻内容，分析新闻文章的情感（负面/中性/正面）。
只输出情感类型，若新闻未提及任何主题相关内容。情感包括：负面, 中性, 正面。
情感的判断如下：
1. 正面：新闻内容若主要包含积极、肯定、赞扬的表达，则可判定为正面情感。例如："某公司成功研发出新型产品，市场反响良好。"
2. 中性：新闻内容若只是客观陈述事实，不带有明显的情感偏向，即为中性情感。例如："今天的气温是25摄氏度。"
3. 负面：若新闻内容包含消极、否定、批判的表达，则为负面情感。例如："某企业因违规操作被处罚。"
请确保输出格式正确，无需其他解释。

新闻内容为："{}"
请你分析新闻的情感。"""

# 定义主题分类和系统提示
TOPICS = [
    "体育", "娱乐", "科技", "时政", "财经", "社会", "国际", "军事", "教育", "生活", "时尚"
]

# 修改主题分类提示，强制在 11 个主题中选择
TOPIC_PROMPT = """你是一个专业的新闻分类专家。请根据新闻内容，从以下 11 个主题中选择一个合适的主题进行分类，一条新闻只能匹配一个主题。
只输出主题名称，不要有其他多余表述。主题包括：体育, 娱乐, 科技, 时政, 财经, 社会, 国际, 军事, 教育, 生活, 时尚。
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

新闻内容为："{}"
请你进行主题分类。"""

# ================== 配置参数 ==================
CACHE_DIR = "cache_data"  # 统一缓存目录
HISTORY_EMB_CACHE = os.path.join(CACHE_DIR, "history_embeddings.pkl")  # 历史文章向量缓存
KEYWORD_CACHE = os.path.join(CACHE_DIR, "keywords_cache.pkl")  # 关键词缓存
BATCH_SIZE = 32  # 批量处理大小
NUM_WORKERS = 4  # 并行工作线程数

# ================== 初始化模型 ==================
# 中文BERT模型（关键词提取）
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese').eval().to(device)

# Sentence-BERT模型（相似度计算）
ST_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
st_tokenizer = AutoTokenizer.from_pretrained(ST_MODEL_NAME)
st_model = AutoModel.from_pretrained(ST_MODEL_NAME).eval().to(device)

# ================== 数据预处理优化 ==================
# 预加载停用词
STOPWORDS = {'的', '了', '在', '是', '和', '等', '为', '对', '并', '这', '就', '也', '都', '一个'}


def preprocess_text(text):
    """预处理文本的优化实现"""
    return re.sub(r'\s+', '', text).replace('\n', '').replace('\t', '')


# ================== 带缓存的批处理关键词提取 ==================
def batch_process_texts(texts, batch_size=32):
    """批量处理文本的分块"""
    all_chunks = []
    for text in texts:
        text = preprocess_text(text)
        chunks = [text]  # 简化处理，不再分块
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


# ================== 历史文章预加载 ==================
def load_history_data(history_file):
    """加载并预处理历史数据"""
    df = pd.read_excel(history_file)  # 从Excel文件读取数据
    df = df.drop_duplicates(subset=['新闻内容'])

    # 检查列名
    print(f"加载的历史数据列名: {df.columns.tolist()}")
    
    # 预生成所有历史文章的向量
    if not os.path.exists(HISTORY_EMB_CACHE):
        print(f"正在预计算历史文章向量...共 {len(df)} 条数据")
        
        # 步骤1: 提取关键词（使用批处理提高效率）
        print("步骤1/3: 批量提取关键词")
        texts = df['新闻内容'].tolist()
        # 使用原始的批量处理方法
        all_keywords = batch_get_keywords(texts)
        print(f"关键词提取完成，共处理 {len(all_keywords)} 条数据")
        
        # 步骤2: 关键词合并
        print("步骤2/3: 合并关键词")
        keyword_texts = []
        for i, kws in enumerate(tqdm(all_keywords, desc="关键词合并")):
            keyword_texts.append("".join(kws))
        print(f"关键词合并完成，共 {len(keyword_texts)} 条数据")
        
        # 步骤3: 计算嵌入向量
        print("步骤3/3: 计算嵌入向量（批量处理）")
        embeddings = []
        total_batches = (len(keyword_texts) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in tqdm(range(0, len(keyword_texts), BATCH_SIZE), desc="计算嵌入向量", total=total_batches):
            batch = keyword_texts[i:i + BATCH_SIZE]
            batch_embeddings = batch_text_embedding(batch)
            embeddings.append(batch_embeddings)
            batch_end = min(i + BATCH_SIZE, len(keyword_texts))
            if (i+BATCH_SIZE) % (BATCH_SIZE*5) == 0 or batch_end == len(keyword_texts):
                print(f"已处理 {batch_end}/{len(keyword_texts)} 条数据 ({batch_end/len(keyword_texts)*100:.1f}%)")
        
        # 合并所有嵌入向量
        all_embeddings = np.vstack(embeddings)
        print(f"嵌入向量计算完成，形状: {all_embeddings.shape}")
        
        # 保存到DataFrame
        df['embedding'] = all_embeddings.tolist()
        
        # 保存缓存
        print(f"保存嵌入向量到缓存: {HISTORY_EMB_CACHE}")
        df.to_pickle(HISTORY_EMB_CACHE)
        print("缓存保存完成!")
    else:
        print(f"从缓存加载历史文章向量: {HISTORY_EMB_CACHE}")
        df = pd.read_pickle(HISTORY_EMB_CACHE)
        print(f"已加载 {len(df)} 条历史文章向量")
    
    print(f"数据列名: {df.columns.tolist()}")
    
    # 确保列名一致性
    if 'topic' not in df.columns:
        print("警告: 历史数据中缺少 'topic' 列，使用'主题'列")
        if '主题' in df.columns:
            print("使用'主题'列作为'topic'列")
            df['topic'] = df['主题']
        else:
            df['topic'] = '社会'  # 添加默认主题
            print("添加默认主题'社会'")
    
    if 'sentiment' not in df.columns:
        print("警告: 历史数据中缺少 'sentiment' 列，使用'分类结果'列")
        if '分类结果' in df.columns:
            print("使用'分类结果'列作为'sentiment'列")
            df['sentiment'] = df['分类结果']
        else:
            df['sentiment'] = '中性'  # 添加默认情感
            print("添加默认情感'中性'")
            
    # 确保content字段存在，用于代码其他部分
    if '新闻内容' in df.columns and 'content' not in df.columns:
        df['content'] = df['新闻内容']
        print("添加'content'列，复制自'新闻内容'列")

    return df


def analyze_news(news):
    """使用LLMClient分析新闻情感和主题"""
    # 分析情感
    sentiment_prompt = SENTIMENT_PROMPT.format(news)
    sentiment_response = llm_client.ask(sentiment_prompt)
    sentiment_result = sentiment_response["answer"].strip()
    
    if sentiment_result not in SENTIMENTS:
        # 增加更严格的后处理，若不匹配则尝试根据关键词判断
        for keyword in ["积极", "肯定", "赞扬"]:
            if keyword in news:
                sentiment_result = "正面"
                break
        else:
            for keyword in ["消极", "否定", "批判"]:
                if keyword in news:
                    sentiment_result = "负面"
                    break
            else:
                sentiment_result = "中性"
    
    # 分析主题
    topic_prompt = TOPIC_PROMPT.format(news)
    topic_response = llm_client.ask(topic_prompt)
    topic_result = topic_response["answer"].strip()
    
    # 若输出不在 11 个主题中，默认选择社会
    if topic_result not in TOPICS:
        topic_result = "社会"
    
    print(f"情感分析结果: {sentiment_result}")
    print(f"主题分类结果: {topic_result}")
    
    return sentiment_result, topic_result


# ================== 优化的推荐函数 ==================
def recommend_articles(input_title, input_content, db_session=None, News=None, top_n=5, use_db=False, history_file="news_history.xlsx"):
    """优化后的推荐函数，支持直接从数据库或Excel文件加载历史数据"""
    # 合并标题和内容
    input_news = f"{input_title} {input_content}"
    
    # 分析情感和主题
    sentiment_result, topic_result = analyze_news(input_news)
    
    if use_db and db_session and News:
        # 从数据库加载历史数据
        print("正在从数据库加载历史新闻...")
        history_news = db_session.query(News).all()
        
        # 构建DataFrame
        data = []
        for news in history_news:
            try:
                # 解析嵌入向量
                embedding = pickle.loads(news.embedding) if news.embedding else None
                
                data.append({
                    'id': news.id,
                    'title': news.title,
                    '新闻内容': news.content,
                    'topic': news.topic,
                    'sentiment': news.sentiment,
                    'embedding': embedding
                })
            except Exception as e:
                print(f"处理新闻ID {news.id} 时出错: {e}")
                continue
        
        history_df = pd.DataFrame(data)
        print(f"从数据库加载了 {len(history_df)} 条历史新闻")
    else:
        # 从Excel文件加载历史数据
        print(f"正在从文件 {history_file} 加载历史新闻...")
        history_df = load_history_data(history_file)
    
    # 确保有历史数据
    if len(history_df) == 0:
        print("警告: 没有可用的历史新闻数据")
        return []
    
    # 清除输入文章的关键词缓存
    if os.path.exists(KEYWORD_CACHE):
        cache = pd.read_pickle(KEYWORD_CACHE)
        if len(cache) > 0:
            cache = cache[:-1]
            pd.to_pickle(cache, KEYWORD_CACHE)
    
    # 处理输入文章
    input_keywords = batch_get_keywords([input_news])[0]
    
    # 清除输入文章的向量缓存
    hash_str = hashlib.md5("".join(input_keywords).encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"batch_{hash_str}.npy")
    if os.path.exists(cache_path):
        os.remove(cache_path)
    
    input_emb = batch_text_embedding(["".join(input_keywords)])[0]
    
    # 检查历史数据中是否所有记录都有嵌入向量
    has_embeddings = all(isinstance(emb, (list, np.ndarray)) for emb in history_df['embedding'])
    
    if not has_embeddings:
        print("警告: 历史数据中有记录缺少嵌入向量，将重新计算")
        # 仅处理没有嵌入向量的记录
        missing_indices = [i for i, emb in enumerate(history_df['embedding']) if not isinstance(emb, (list, np.ndarray))]
        if missing_indices:
            # 使用正确的列名获取文本
            text_col = '新闻内容' if '新闻内容' in history_df.columns else 'content'
            texts = history_df.iloc[missing_indices][text_col].tolist()
            keywords = batch_get_keywords(texts)
            keyword_texts = ["".join(kws) for kws in keywords]
            embeddings = batch_text_embedding(keyword_texts)
            
            for idx, emb in zip(missing_indices, embeddings):
                history_df.at[idx, 'embedding'] = emb
    
    # 批量计算相似度
    history_embs = np.array(history_df['embedding'].tolist())
    similarities = cosine_similarity([input_emb], history_embs)[0]
    
    # 获取主题和情感列名
    topic_col = 'topic'
    sentiment_col = 'sentiment'
    
    # 主题匹配得分
    theme_scores = [1 if str(history_df.iloc[i][topic_col]) == topic_result else 0 for i in range(len(history_df))]
    
    # 情感匹配得分
    sentiment_scores = [1 if str(history_df.iloc[i][sentiment_col]) == sentiment_result else 0 for i in range(len(history_df))]
    
    # 综合得分
    combined_scores = [0.4 * s + 0.3 * t + 0.3 * k for s, t, k in zip(similarities, theme_scores, sentiment_scores)]
    
    # 获取TopN结果
    top_indices = np.array(combined_scores).argsort()[-top_n:][::-1]
    results = []
    
    for idx in top_indices:
        row = history_df.iloc[idx]
        
        # 使用一致的列名
        content_col = '新闻内容' if '新闻内容' in row.index else 'content'
        id_col = 'id' if 'id' in row.index else 'ID'
        
        result_item = {
            "id": int(row[id_col]) if id_col in row else idx,
            "content": str(row[content_col]) if content_col in row else "",
            "theme": str(row[topic_col]) if topic_col in row else topic_result,
            "label": str(row[sentiment_col]) if sentiment_col in row else sentiment_result,
            "similarity": round(float(combined_scores[idx]), 4)
        }
        
        # 添加标题字段（如果有）
        if 'title' in row.index:
            result_item["title"] = str(row['title'])
        
        results.append(result_item)
    
    return results


# ================== 使用示例 ==================
if __name__ == "__main__":
    # 初始化缓存目录
    os.makedirs(CACHE_DIR, exist_ok=True)

    sample_news = """臨近國慶節，香港中環街頭掛滿了慶祝中華人民共和國成立75周年標語，營造濃厚的慶祝國慶氛圍。（香港中通社記者  謝光磊 攝）  香港中通社圖片"""
    recommendations = recommend_articles("", sample_news)

    print(f"输入文章: {sample_news}")
    print("\n推荐历史文章TOP5:")
    for i, item in enumerate(recommendations, 1):
        print(f"{i}. [相似度{item['similarity']:.4f}] {item['content'][:50]}...")