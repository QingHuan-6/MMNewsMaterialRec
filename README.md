## 后端flask配置

1.pip install elasticsearch==8.11.1

在app/config.py下修改数据库配置文件

```python
1. SERVER_IP
2.SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}?charset=utf8mb4&init_command=SET%20time_zone%3D%27%2B08%3A00%27"
   
```


## elasticsearch配置

1. 安装运行Elasticsearch 服务器`docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:8.11.1`
2. 进入docker容器配置elasticsearch服务的user和密码
    docker exec -it elasticsearch /bin/bash
    /usr/share/elasticsearch/bin/elasticsearch-setup-passwords auto
3. 修改flask后端app/utils/es_utils.py账号密码
4. 运行python reset_indices.py  完成索引创建
5. curl -X POST http://localhost:5000/api/admin/migrate-vectors 完成图片向量迁移(运行python run.py后)
6. curl -X POST http://localhost:5000/api/news/migrate-vectors   完成新闻向量迁移(运行python run.py后)


## mysql新增列:

重新写入sql