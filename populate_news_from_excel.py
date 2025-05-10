import os
import pandas as pd
import pickle
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import db, News  # 假设你的 Flask app 和 SQLAlchemy db 对象已正确配置并可导入
from novo2 import batch_get_keywords, batch_text_embedding # 从 novo2.py 导入所需函数

# 数据库连接配置 (根据你的实际情况修改)
DATABASE_URI = 'mysql+pymysql://root:123456@localhost/pixtock'

def populate_news_from_excel(excel_path="news_history.xlsx"):
    """
    清空 news 表，从 Excel 文件读取新闻数据，计算嵌入向量，并存入数据库。

    参数:
        excel_path (str): 包含新闻数据的 Excel 文件路径。
                          需要包含 '新闻内容', '主题', '情感' 列。
                          可选 '标题' 列。
    """
    engine = create_engine(DATABASE_URI)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db_session = SessionLocal()

    try:
        # 1. 清空 news 数据库表
        print("正在清空 news 表...")
        num_deleted = db_session.query(News).delete()
        db_session.commit()
        print(f"成功删除 {num_deleted} 条旧新闻记录。")

        # 2. 读取 news_history.xlsx 文件
        print(f"正在从 {excel_path} 读取新闻数据...")
        if not os.path.exists(excel_path):
            print(f"错误: Excel 文件 {excel_path} 未找到。")
            return

        try:
            df = pd.read_excel(excel_path)
        except Exception as e:
            print(f"读取 Excel 文件时出错: {e}")
            return
        
        # 检查必需的列是否存在
        required_columns = ['新闻文本', '主题', '情感']
        for col in required_columns:
            if col not in df.columns:
                print(f"错误: Excel 文件中缺少必需的列 '{col}'。")
                return
        
        print(f"成功读取 {len(df)} 条新闻数据。")

        # 3. & 4. 计算嵌入向量并存储到数据库
        print("开始处理新闻并存入数据库...")
        processed_count = 0
        failed_count = 0

        for index, row in df.iterrows():
            try:
                title = str(row.get('标题', '')) # 如果没有标题列，则为空字符串
                content = str(row['新闻文本'])
                topic = str(row['主题'])
                sentiment = str(row['情感'])

                if not content: # 跳过内容为空的行
                    print(f"警告: 第 {index+2} 行新闻内容为空，已跳过。")
                    failed_count +=1
                    continue

                # 创建文本嵌入向量
                text_for_embedding = f"{title} {content}"
                embedding_pickle = None # 初始化
                try:
                    # 提取关键词
                    keywords = batch_get_keywords([text_for_embedding])[0]
                    keyword_text = "".join(keywords)
                    
                    # 生成嵌入向量
                    if keyword_text: # 确保关键词文本不为空
                        embedding_array = batch_text_embedding([keyword_text])[0]
                        embedding_list = embedding_array.tolist()
                        embedding_pickle = pickle.dumps(embedding_list)
                    else:
                        print(f"警告: 第 {index+2} 行提取关键词为空，嵌入向量将为 None。")

                except Exception as e:
                    print(f"计算新闻 (第 {index+2} 行) 的嵌入向量时出错: {e}")
                    # embedding_pickle 保持为 None

                # 创建新闻对象
                new_news_item = News(
                    title=title,
                    content=content,
                    topic=topic,
                    sentiment=sentiment,
                    embedding=embedding_pickle,
                    upload_time=datetime.now()
                )
                
                db_session.add(new_news_item)
                processed_count += 1

                # 为了避免内存占用过大和事务过长，可以分批提交
                if (index + 1) % 100 == 0:
                    db_session.commit()
                    print(f"已处理并提交 {processed_count} 条新闻...")

            except Exception as e:
                print(f"处理新闻 (第 {index+2} 行) 时发生错误: {e}")
                failed_count += 1
                db_session.rollback() # 如果单条出错，回滚当前未提交的更改
                # 可以选择继续处理下一条，或者在此处中断

        db_session.commit() # 提交剩余的记录
        print(f"所有新闻处理完毕。成功处理: {processed_count} 条, 失败: {failed_count} 条。")

    except Exception as e:
        print(f"执行过程中发生严重错误: {e}")
        if db_session:
            db_session.rollback()
    finally:
        if db_session:
            db_session.close()

if __name__ == "__main__":
    # 注意：直接运行此脚本会修改数据库。
    # 确保备份数据或在测试环境中运行。
    
    # 假设你的 Flask app 和 SQLAlchemy db 对象已经初始化
    # 如果在 Flask 应用上下文之外运行，可能需要手动初始化 SQLAlchemy
    # 例如:
    # from flask import Flask
    # app = Flask(__name__)
    # app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
    # app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # db.init_app(app)
    # with app.app_context():
    #     populate_news_from_excel()
    #
    # 或者直接使用上面创建的 engine 和 session
    
    print("开始执行新闻数据填充脚本...")
    # 这里提供一个基本的执行方式，但强烈建议在Flask应用上下文中执行，以确保db对象正确初始化
    # 如果直接运行，请确保 models.py 中的 db 对象能被正确使用。
    # 你可能需要调整 `models.py` 的导入或此处的初始化方式。
    
    # 示例：直接调用 (需要确保数据库连接和模型可用)
    # 确保 `news_history.xlsx` 文件与此脚本在同一目录下或提供正确路径
    populate_news_from_excel(excel_path="news_history.xlsx") # 修改为实际路径
    print("脚本执行完毕。") 