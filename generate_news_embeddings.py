import os
import numpy as np
import pickle
import torch
from flask import Flask
from models import db, News
from tqdm import tqdm
from novo2 import batch_get_keywords, batch_text_embedding, preprocess_text
import hashlib
import pandas as pd

# 确保缓存目录存在
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

# 历史嵌入向量缓存文件
HISTORY_EMB_CACHE = os.path.join(CACHE_DIR, "history_embeddings.pkl")

# 批处理大小
BATCH_SIZE = 32
# 并行工作线程数
NUM_WORKERS = 4

# 初始化Flask应用
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost/pixtock'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def clear_news_embeddings():
    """清空所有新闻的嵌入向量"""
    with app.app_context():
        print("正在清空所有新闻的嵌入向量...")
        # 查询所有新闻
        all_news = News.query.all()
        total_news = len(all_news)
        
        if total_news == 0:
            print("数据库中没有新闻记录")
            return False
        
        # 清空所有新闻的嵌入向量
        cleared_count = 0
        for news in tqdm(all_news, desc="清空嵌入向量"):
            if news.embedding is not None:
                news.embedding = None
                cleared_count += 1
        
        # 提交更改
        db.session.commit()
        print(f"成功清空了 {cleared_count}/{total_news} 条新闻的嵌入向量")
        return True

def generate_news_embeddings():
    """为数据库中的所有新闻生成嵌入向量，采用novo22.py的三阶段处理方式"""
    print("开始为新闻数据生成嵌入向量...")
    
    with app.app_context():
        # 查询所有没有嵌入向量的新闻
        news_without_embedding = News.query.filter(News.embedding == None).all()
        total_news = len(news_without_embedding)
        
        if total_news == 0:
            print("没有找到需要生成嵌入向量的新闻记录")
            return
            
        print(f"找到 {total_news} 条新闻需要生成嵌入向量")
        
        # 准备文本内容
        contents = []
        news_ids = []
        for news in news_without_embedding:
            # 如果有标题，合并标题和内容
            if news.title:
                full_text = f"{news.title}。{news.content}"
            else:
                full_text = news.content
            contents.append(preprocess_text(full_text))
            news_ids.append(news.id)
        
        # 步骤1: 提取关键词（使用批处理提高效率）
        print(f"\n步骤1/3: 批量提取关键词 (共{total_news}条)")
        
        # 生成缓存文件名 - 基于内容的哈希，保证重复内容可以复用缓存
        cache_hash = hashlib.md5(str(news_ids).encode()).hexdigest()
        keywords_cache_file = os.path.join(CACHE_DIR, f"news_keywords_{cache_hash}.pkl")
        
        if os.path.exists(keywords_cache_file):
            print(f"从缓存加载关键词: {keywords_cache_file}")
            with open(keywords_cache_file, 'rb') as f:
                all_keywords = pickle.load(f)
        else:
            # 分批处理关键词提取
            all_keywords = []
            for i in range(0, total_news, BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, total_news)
                batch = contents[i:batch_end]
                print(f"提取批次 {i//BATCH_SIZE + 1}/{(total_news-1)//BATCH_SIZE + 1} 的关键词 ({i+1}-{batch_end}/{total_news})")
                batch_keywords = batch_get_keywords(batch)
                all_keywords.extend(batch_keywords)
                # 显示进度
                print(f"已完成 {batch_end}/{total_news} 条 ({batch_end/total_news*100:.1f}%)")
            
            # 保存关键词缓存
            with open(keywords_cache_file, 'wb') as f:
                pickle.dump(all_keywords, f)
                
        print(f"关键词提取完成，共处理 {len(all_keywords)} 条数据")
        
        # 步骤2: 关键词合并
        print("\n步骤2/3: 合并关键词")
        keyword_texts = []
        for i, kws in enumerate(tqdm(all_keywords, desc="关键词合并")):
            keyword_texts.append("".join(kws))
            if (i+1) % (BATCH_SIZE*2) == 0 or i+1 == len(all_keywords):
                print(f"已处理 {i+1}/{len(all_keywords)} 条数据 ({(i+1)/len(all_keywords)*100:.1f}%)")
        print(f"关键词合并完成，共 {len(keyword_texts)} 条")
        
        # 步骤3: 计算嵌入向量
        print("\n步骤3/3: 计算嵌入向量（批量处理）")
        embeddings = []
        total_batches = (len(keyword_texts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(0, len(keyword_texts), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(keyword_texts))
            batch = keyword_texts[i:batch_end]
            
            # 生成当前批次的缓存标识
            batch_hash = hashlib.md5("|".join(batch).encode()).hexdigest()
            cache_path = os.path.join(CACHE_DIR, f"news_batch_{batch_hash}.npy")
            
            if os.path.exists(cache_path):
                print(f"从缓存加载批次 {i//BATCH_SIZE + 1}/{total_batches} 的嵌入向量")
                batch_embeddings = np.load(cache_path)
            else:
                print(f"计算批次 {i//BATCH_SIZE + 1}/{total_batches} 的嵌入向量 ({i+1}-{batch_end}/{len(keyword_texts)})")
                batch_embeddings = batch_text_embedding(batch)
                # 保存缓存
                np.save(cache_path, batch_embeddings)
            
            embeddings.append(batch_embeddings)
            print(f"已处理 {batch_end}/{len(keyword_texts)} 条数据 ({batch_end/len(keyword_texts)*100:.1f}%)")
        
        # 合并所有嵌入向量
        all_embeddings = np.vstack(embeddings)
        print(f"\n嵌入向量计算完成，形状: {all_embeddings.shape}")
        
        # 保存到数据库
        print("\n正在将嵌入向量保存到数据库...")
        for i, news_id in enumerate(tqdm(news_ids, desc="保存到数据库")):
            # 查询新闻记录
            news = News.query.get(news_id)
            if news:
                # 将numpy数组序列化为二进制数据
                news.embedding = pickle.dumps(all_embeddings[i])
            
            # 每100条提交一次，避免事务过大
            if (i + 1) % 100 == 0 or i == len(news_ids) - 1:
                db.session.commit()
                print(f"已保存 {i+1}/{len(news_ids)} 条记录 ({(i+1)/len(news_ids)*100:.1f}%)")
        
        print("\n所有新闻嵌入向量生成并保存完成！")

def regenerate_all_embeddings():
    """清空并重新生成所有新闻的嵌入向量"""
    print("开始重新生成所有新闻的嵌入向量...")
    
    # 第1步：清空所有嵌入向量
    if clear_news_embeddings():
        # 第2步：重新生成所有嵌入向量
        generate_news_embeddings()
        print("所有新闻的嵌入向量已成功重新生成！")
    else:
        print("没有需要处理的新闻记录")

def import_embeddings_from_cache():
    """从缓存的历史嵌入向量文件导入向量到数据库中的新闻记录"""
    print(f"开始从历史文件 {HISTORY_EMB_CACHE} 导入嵌入向量到数据库...")
    
    if not os.path.exists(HISTORY_EMB_CACHE):
        print(f"错误：历史嵌入向量缓存文件 {HISTORY_EMB_CACHE} 不存在！")
        return False
    
    try:
        # 加载历史嵌入向量缓存
        history_df = pd.read_pickle(HISTORY_EMB_CACHE)
        print(f"成功加载历史嵌入向量文件，包含 {len(history_df)} 条记录")
        
        # 打印DataFrame的列名，用于调试
        print(f"历史文件的列名：{history_df.columns.tolist()}")
        
        # 确保数据有需要的列
        content_col = '新闻内容'
        if content_col not in history_df.columns:
            if 'content' in history_df.columns:
                content_col = 'content'
            elif '新闻文本' in history_df.columns:
                content_col = '新闻文本'
            else:
                print(f"错误：找不到内容列，可用列：{history_df.columns.tolist()}")
                return False
        
        # 确保存在嵌入向量列
        if 'embedding' not in history_df.columns:
            print("错误：历史文件中没有'embedding'列")
            return False
        
        # 显示向量维度信息
        if len(history_df) > 0:
            first_vec = history_df['embedding'].iloc[0]
            print(f"历史文件中的向量维度: {len(first_vec)}")
        
        with app.app_context():
            # 查询数据库中的所有新闻记录
            all_news = News.query.all()
            print(f"数据库中有 {len(all_news)} 条新闻记录")
            
            # 创建内容到向量的映射
            content_to_embedding = {}
            for i, row in tqdm(history_df.iterrows(), total=len(history_df), desc="创建内容映射"):
                content = row[content_col]
                embedding = row['embedding']
                # 可以对内容进行预处理，使其更容易匹配
                content = preprocess_text(content)
                content_to_embedding[content] = embedding
            
            print(f"创建了 {len(content_to_embedding)} 条内容到向量的映射")
            
            # 为数据库中的新闻记录匹配并导入向量
            matched_count = 0
            not_matched_count = 0
            
            for news in tqdm(all_news, desc="更新数据库记录"):
                # 预处理新闻内容
                news_content = preprocess_text(news.content)
                
                # 尝试直接匹配
                if news_content in content_to_embedding:
                    # 找到匹配，更新向量
                    news.embedding = pickle.dumps(content_to_embedding[news_content])
                    matched_count += 1
                else:
                    # 如果不能直接匹配，尝试查找最相似的内容
                    best_match = None
                    best_score = 0
                    
                    for cache_content in content_to_embedding.keys():
                        # 使用简单的字符串包含关系计算相似度
                        # 这个方法可以根据需要改进
                        if news_content in cache_content or cache_content in news_content:
                            # 计算重叠比例
                            overlap = min(len(news_content), len(cache_content)) / max(len(news_content), len(cache_content))
                            if overlap > best_score:
                                best_score = overlap
                                best_match = cache_content
                    
                    # 如果找到足够好的匹配（超过80%相似度）
                    if best_match and best_score > 0.8:
                        news.embedding = pickle.dumps(content_to_embedding[best_match])
                        matched_count += 1
                    else:
                        not_matched_count += 1
                
                # 每100条提交一次
                if (matched_count + not_matched_count) % 100 == 0:
                    db.session.commit()
                    print(f"已处理: {matched_count + not_matched_count}/{len(all_news)}, 匹配: {matched_count}, 未匹配: {not_matched_count}")
            
            # 最终提交
            db.session.commit()
            
            print(f"\n导入完成！总共处理 {len(all_news)} 条新闻记录")
            print(f"成功匹配并导入向量: {matched_count} 条")
            print(f"未找到匹配: {not_matched_count} 条")
            
            if not_matched_count > 0:
                print("\n对于未匹配的记录，建议使用generate_news_embeddings()函数生成向量")
            
            return matched_count > 0
            
    except Exception as e:
        print(f"导入过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 调用导入函数，从历史缓存导入向量
    if import_embeddings_from_cache():
        print("\n从历史缓存导入向量成功！")
        
        # 询问是否为未匹配的记录生成向量
        print("\n是否要为未匹配的记录生成新的嵌入向量? (y/n)")
        choice = input().strip().lower()
        if choice == 'y' or choice == 'yes':
            generate_news_embeddings()
    else:
        print("\n从历史缓存导入向量失败，将使用标准方法重新生成所有向量")
        regenerate_all_embeddings() 