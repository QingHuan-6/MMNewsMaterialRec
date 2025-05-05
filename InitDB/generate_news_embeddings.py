import os
import numpy as np
import pickle
import torch
from flask import Flask
from models import db, News
from tqdm import tqdm
from novo2 import batch_get_keywords, batch_text_embedding, preprocess_text

# 确保缓存目录存在
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

# 批处理大小
BATCH_SIZE = 32

# 初始化Flask应用
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost/pixtock'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def generate_news_embeddings():
    """为数据库中的所有新闻生成嵌入向量"""
    print("开始为新闻数据生成嵌入向量...")
    
    with app.app_context():
        # 查询所有没有嵌入向量的新闻
        news_without_embedding = News.query.filter(News.embedding == None).all()
        total_news = len(news_without_embedding)
        
        if total_news == 0:
            print("没有找到需要生成嵌入向量的新闻记录")
            return
            
        print(f"找到 {total_news} 条新闻需要生成嵌入向量")
        
        # 分批处理，避免内存溢出
        for i in range(0, total_news, BATCH_SIZE):
            batch_news = news_without_embedding[i:i+BATCH_SIZE]
            print(f"处理批次 {i//BATCH_SIZE + 1}/{(total_news-1)//BATCH_SIZE + 1}，共 {len(batch_news)} 条新闻")
            
            # 准备文本内容
            contents = []
            for news in batch_news:
                # 如果有标题，合并标题和内容
                if news.title:
                    full_text = f"{news.title}。{news.content}"
                else:
                    full_text = news.content
                contents.append(preprocess_text(full_text))
            
            # 提取关键词
            print("提取关键词...")
            keywords_list = batch_get_keywords(contents)
            
            # 生成嵌入向量
            print("生成嵌入向量...")
            keyword_texts = ["".join(keywords) for keywords in keywords_list]
            embeddings = batch_text_embedding(keyword_texts)
            
            # 将嵌入向量保存到数据库
            print("保存嵌入向量到数据库...")
            for j, news in enumerate(batch_news):
                # 将numpy数组序列化为二进制数据
                news.embedding = pickle.dumps(embeddings[j])
            
            # 提交到数据库
            db.session.commit()
            print(f"成功更新了 {len(batch_news)} 条新闻的嵌入向量")
        
        print("所有新闻嵌入向量生成完成！")

if __name__ == "__main__":
    generate_news_embeddings() 