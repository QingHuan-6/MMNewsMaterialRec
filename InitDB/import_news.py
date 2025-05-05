import pandas as pd
from flask import Flask
from models import db, News
import os
from datetime import datetime

# 创建Flask应用
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost/pixtock'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def import_news_from_excel(excel_file='news_history.xlsx'):
    """从Excel文件导入新闻数据到数据库"""
    print(f"开始从 {excel_file} 导入新闻数据...")
    
    # 检查文件是否存在
    if not os.path.exists(excel_file):
        print(f"错误: 文件 {excel_file} 不存在!")
        return
    
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_file)
        print(f"成功读取文件，共有 {len(df)} 条新闻记录")
        
        # 检查必要的列是否存在
        required_columns = ['新闻内容', '主题', '分类结果']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"错误: Excel文件缺少必要的列: {', '.join(missing_columns)}")
            return
        
        # 设置主题和情感映射
        topic_mapping = {
            '体育': '体育', '体育新闻': '体育', 
            '娱乐': '娱乐', '娱乐新闻': '娱乐',
            '科技': '科技', '科技新闻': '科技',
            '时政': '时政', '政治': '时政', '政治新闻': '时政',
            '财经': '财经', '经济': '财经', '财经新闻': '财经',
            '社会': '社会', '社会新闻': '社会',
            '国际': '国际', '国际新闻': '国际', '国际关系': '国际',
            '军事': '军事', '军事新闻': '军事',
            '教育': '教育', '教育新闻': '教育',
            '生活': '生活', '日常生活': '生活', '生活新闻': '生活',
            '时尚': '时尚', '时尚新闻': '时尚'
        }
        
        sentiment_mapping = {
            '正面': '正面', '积极': '正面', '好': '正面', '正': '正面', 
            '中性': '中性', '中立': '中性', '中': '中性',
            '负面': '负面', '消极': '负面', '坏': '负面', '负': '负面'
        }
        
        with app.app_context():
            # 确保表存在
            db.create_all()
            
            # 检查是否已有新闻数据
            existing_count = News.query.count()
            if existing_count > 0:
                print(f"警告: 数据库中已有 {existing_count} 条新闻记录。继续导入将添加新记录而不覆盖。")
                confirm = input("是否继续导入? (y/n): ")
                if confirm.lower() != 'y':
                    print("导入取消")
                    return
            
            # 导入数据
            success_count = 0
            error_count = 0
            
            for _, row in df.iterrows():
                try:
                    # 获取并清洗数据 - 不需要标题字段
                    content = str(row.get('新闻内容', '')).strip()
                    
                    # 检查内容是否为空
                    if not content or pd.isna(content):
                        print(f"警告: 跳过ID为 {_} 的记录，内容为空")
                        error_count += 1
                        continue
                    
                    # 获取主题并映射
                    raw_topic = str(row.get('主题', '')).strip()
                    if raw_topic in topic_mapping:
                        topic = topic_mapping[raw_topic]
                    else:
                        # 如果不在映射中，使用其他作为默认值或者根据需要处理
                        print(f"警告: ID为 {_} 的记录主题 '{raw_topic}' 不在预定义类别中，将使用 '其他' 作为主题")
                        topic = '其他'
                    
                    # 获取情感分类并映射
                    raw_sentiment = str(row.get('分类结果', '')).strip()
                    if raw_sentiment in sentiment_mapping:
                        sentiment = sentiment_mapping[raw_sentiment]
                    else:
                        # 默认为中性
                        print(f"警告: ID为 {_} 的记录情感分类 '{raw_sentiment}' 不在预定义类别中，将使用 '中性' 作为默认值")
                        sentiment = '中性'
                    
                    # 创建新闻记录 - 标题设为None
                    news = News(
                        title=None,
                        content=content,
                        topic=topic,
                        sentiment=sentiment,
                        upload_time=datetime.now()
                    )
                    
                    # 添加到数据库
                    db.session.add(news)
                    success_count += 1
                    
                    # 每100条提交一次，减少内存使用
                    if success_count % 100 == 0:
                        db.session.commit()
                        print(f"已导入 {success_count} 条记录...")
                    
                except Exception as e:
                    error_count += 1
                    print(f"导入ID为 {_} 的记录时出错: {e}")
            
            # 最终提交
            try:
                db.session.commit()
                print(f"导入完成! 成功导入 {success_count} 条记录，失败 {error_count} 条记录")
            except Exception as e:
                db.session.rollback()
                print(f"提交数据时发生错误: {e}")
                print(f"导入失败! 成功导入 0 条记录，失败 {success_count + error_count} 条记录")
    
    except Exception as e:
        print(f"导入过程中发生错误: {e}")

if __name__ == "__main__":
    print("新闻数据导入工具")
    excel_file = input("请输入Excel文件路径 (默认为 news_history.xlsx): ").strip() or 'news_history.xlsx'
    import_news_from_excel(excel_file) 