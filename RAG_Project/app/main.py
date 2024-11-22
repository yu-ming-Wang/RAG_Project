from pymongo import MongoClient
from qdrant_client import QdrantClient
import os

#27017 是 MongoDB 的 默認通訊端口
#我的 main.py 是在 app 容器內運行，因此它可以通過 Docker 的內部 DNS
#直接用 mongodb:27017 與 MongoDB 容器通信，而不需要使用宿主機的地址（如 localhost）。
mongo_client = MongoClient("mongodb://mongodb:27017/")
db = mongo_client['rag_database']
print("Connected to MongoDB!")

qdrant_client = QdrantClient("qdrant", port=6333)
print("Connected to Qdrant!")

print("Connections established successfully!")