#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask
from models import db, News
import time

# 设置工作目录和路径
CACHE_DIR = "cache_data"
HISTORY_EMB_CACHE = os.path.join(CACHE_DIR, "history_embeddings.pkl")

# 初始化Flask应用（用于数据库连接）
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost/pixtock'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def compare_embedding_dimensions():
    """比较历史文章pkl文件和数据库中新闻向量维度"""
    print("\n====== 开始比较向量维度 ======\n")
    
    # 1. 加载历史文章向量
    if os.path.exists(HISTORY_EMB_CACHE):
        print(f"正在从 {HISTORY_EMB_CACHE} 加载历史文章向量...")
        try:
            history_df = pd.read_pickle(HISTORY_EMB_CACHE)
            print(f"历史文章数量: {len(history_df)}")
            
            # 获取第一条向量信息
            if len(history_df) > 0 and 'embedding' in history_df.columns:
                first_vector = history_df['embedding'].iloc[0]
                
                # 提取并显示向量信息
                print("\n----历史向量信息----")
                print(f"向量维度: {np.array(first_vector).shape}")
                print(f"向量类型: {type(first_vector)}")
                print(f"数据类型: {np.array(first_vector).dtype}")
                
                # 序列化并计算大小
                serialized = pickle.dumps(first_vector)
                serialized_size = len(serialized) / 1024  # KB
                print(f"序列化大小: {serialized_size:.2f} KB")
                
                # 检查是否为ndarray
                if isinstance(first_vector, np.ndarray):
                    print(f"是否为ndarray: 是")
                else:
                    print(f"是否为ndarray: 否，转换后维度: {np.array(first_vector).shape}")
                
                # 检查是否使用了压缩
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    size = len(pickle.dumps(first_vector, protocol=protocol)) / 1024
                    print(f"Protocol {protocol} 序列化大小: {size:.2f} KB")
            else:
                print("历史文件中没有找到embedding列或为空")
        except Exception as e:
            print(f"加载历史向量出错: {e}")
    else:
        print(f"历史文章向量文件不存在: {HISTORY_EMB_CACHE}")
    
    # 2. 从数据库加载新闻向量
    with app.app_context():
        try:
            print("\n正在从数据库加载新闻向量...")
            news_count = News.query.count()
            print(f"数据库新闻总数: {news_count}")
            
            # 获取有向量的新闻数量
            news_with_embedding = News.query.filter(News.embedding != None).count()
            print(f"带有向量的新闻数量: {news_with_embedding}")
            
            if news_with_embedding > 0:
                # 获取第一条带有向量的新闻
                all_news = News.query.filter(News.embedding.isnot(None)).all()
                news = all_news[-1] if all_news else None
                if news and news.embedding:
                    # 反序列化向量
                    db_vector = pickle.loads(news.embedding)
                    
                    # 提取并显示向量信息
                    print("\n----数据库向量信息----")
                    print(f"向量维度: {np.array(db_vector).shape}")
                    print(f"向量类型: {type(db_vector)}")
                    print(f"数据类型: {np.array(db_vector).dtype}")
                    
                    # 计算序列化大小
                    serialized_size = len(news.embedding) / 1024  # KB
                    print(f"数据库中存储大小: {serialized_size:.2f} KB")
                    
                    # 重新序列化并计算大小
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                        size = len(pickle.dumps(db_vector, protocol=protocol)) / 1024
                        print(f"Protocol {protocol} 重新序列化大小: {size:.2f} KB")
                    
                    # 比较内容
                    if 'embedding' in locals() and 'first_vector' in locals():
                        db_vector_flat = np.array(db_vector).flatten()
                        hist_vector_flat = np.array(first_vector).flatten()
                        
                        if db_vector_flat.shape == hist_vector_flat.shape:
                            print(f"\n两种向量维度相同: {db_vector_flat.shape}")
                            # 检查前几个值是否相同
                            print(f"数据库向量后10个值: {db_vector_flat[-10:-1]}")
                            print(f"历史向量前10个值: {hist_vector_flat[:10]}")
                        else:
                            print(f"\n两种向量维度不同!")
                            print(f"数据库向量维度: {db_vector_flat.shape}")
                            print(f"历史向量维度: {hist_vector_flat.shape}")
            else:
                print("数据库中没有找到带向量的新闻")
        except Exception as e:
            print(f"加载数据库向量出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n====== 比较完成 ======\n")

if __name__ == "__main__":
    compare_embedding_dimensions()
