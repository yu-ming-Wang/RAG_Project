from pymongo import MongoClient
import gzip
import base64
from typing import Dict, List
from clearml import Task
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from pymongo.errors import ConnectionFailure
import uuid

# Replace tokenizer with Smollm 135
# 步骤1：從 MongoDB 提取和解压數據
def decompress_content(compressed_str: str) -> str:
    """解壓縮base64字符串轉換回的壓縮内容
    """
    try:
        compressed_bytes = base64.b64decode(compressed_str)
        return gzip.decompress(compressed_bytes).decode('utf-8')
    except Exception as e:
        print(f"Failed to decompress content: {e}")
        return ""

#def fetch_and_decompress_data(db_name: str = "llm_twin", collection_name: str = "githubdata_test") -> List[Dict]:
    """從 MongoDB 中提取數據并進行解壓
    """
    try:
        client = MongoClient("mongodb://localhost:27017/")
        client.admin.command('ping')  # 確保 MongoDB 服務器是可用的
    except ConnectionFailure:
        print("無法連接到 MongoDB 數據庫，請確保 MongoDB 服務器正在運行。")
        return []

    db = client[db_name]
    collection = db[collection_name]

    task = Task.init(project_name="Featurization_Pipeline", task_name="Fetch and Decompress Data")
    logger = task.get_logger()

    decompressed_data = []
    for doc in collection.find():
        content = doc.get("content", {})
        if isinstance(content, dict):
            for file_name, file_info in content.items():
                if isinstance(file_info, dict) and file_info.get("type") == "compressed":
                    # 解壓縮內容
                    decompressed_text = decompress_content(file_info["content"])
                    if decompressed_text:
                        decompressed_data.append({"text": decompressed_text, "metadata": doc.get("source", {})})
                    else:
                        logger.report_text(f"Failed to decompress content for file {file_name}")
                elif isinstance(file_info, str):
                    # 如果 `file_info` 是字串，直接添加文本
                    decompressed_data.append({"text": file_info, "metadata": doc.get("source", {})})

    logger.report_text(f"Total documents fetched and decompressed: {len(decompressed_data)}")
    task.close()
    return decompressed_data
from gridfs import GridFS

def fetch_and_decompress_data(db_name: str = "llm_twin", collection_name: str = "") -> List[Dict]:
    """
    從 MongoDB 中提取數據，包括直接存儲的數據和存儲於 GridFS 的大文件，並進行解壓縮。
    """
    try:
        client = MongoClient("mongodb://localhost:27017/")
        client.admin.command('ping')  # 確保 MongoDB 服務器可用
    except ConnectionFailure:
        print("無法連接到 MongoDB，請確保 MongoDB 服務正在運行。")
        return []

    db = client[db_name]
    collection = db[collection_name]
    gridfs = GridFS(db)

    decompressed_data = []
    for doc in collection.find():
        if "gridfs_id" in doc:  # 如果文檔使用了 GridFS
            try:
                gridfs_file = gridfs.get(doc["gridfs_id"])  # 獲取 GridFS 文件
                compressed_content = gridfs_file.read()  # 讀取壓縮數據
                decompressed_text = gzip.decompress(compressed_content).decode('utf-8')  # 解壓縮內容
                decompressed_data.append({
                    "text": decompressed_text,
                    "metadata": doc.get("metadata", {})  # 保留元數據
                })
            except Exception as e:
                print(f"提取或解壓縮 GridFS 文件失敗，ID: {doc['gridfs_id']}, 錯誤: {e}")
        elif "content" in doc and isinstance(doc["content"], dict):  # 直接存儲的數據
            for file_name, file_info in doc["content"].items():
                if file_info.get("type") == "compressed":
                    try:
                        compressed_content = base64.b64decode(file_info["content"])  # 解碼 base64
                        decompressed_text = gzip.decompress(compressed_content).decode('utf-8')  # 解壓縮內容
                        decompressed_data.append({
                            "text": decompressed_text,
                            "metadata": doc.get("source", {})  # 保留元數據
                        })
                    except Exception as e:
                        print(f"解壓縮內容失敗，文件: {file_name}, 錯誤: {e}")
        else:
            print(f"無法處理文檔: {doc['_id']}, 未知數據格式")

    print(f"成功提取並解壓縮 {len(decompressed_data)} 條數據。")
    return decompressed_data


# 步骤2：清理和切分數據
def clean_text(text: str) -> str:
    """清理文本內容，去掉多餘的空格等无用符號"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
    """將文本分割為多個小塊，每個小塊大小為 chunk_size"""
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def clean_and_chunk_data(fetched_data: List[Dict], chunk_size: int = 200) -> List[Dict]:
    """清理和切分數據
    """
    task = Task.init(project_name="Featurization_Pipeline", task_name="Clean and Chunk Data")
    logger = task.get_logger()

    cleaned_chunks = []
    for data in fetched_data:
        # 清理文本
        cleaned_text = clean_text(data["text"])
        logger.report_text(f"清理後的文本內容（前100字符）: {cleaned_text[:100]}...")
        
        # 切分文本
        chunks = chunk_text(cleaned_text, chunk_size)
        logger.report_text(f"切分得到的文本塊數量: {len(chunks)}")
        
        for chunk in chunks:
            cleaned_chunks.append({"text": chunk, "metadata": data["metadata"]})

    task.close()
    return cleaned_chunks

# 步骤3：生成嵌入向量（使用 Smollm 135）
huggingface_token = "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"
model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
# Add padding token to avoid padding error
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name, token=huggingface_token).half().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

def generate_embeddings(chunks: List[Dict]) -> List[Dict]:
    """為每個文本塊生成嵌入，使用 Smollm 135 模型
    """
    embedded_data = []

    task = Task.init(project_name="Featurization_Pipeline", task_name="Generate Embeddings with Smollm 135")
    logger = task.get_logger()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for chunk in chunks:
        # 使用 Smollm 135 生成嵌入
        inputs = tokenizer(chunk["text"], return_tensors="pt", padding=True, truncation=True, max_length=512).to(torch.device(device))        
        with torch.no_grad():                                    
            outputs = model(**inputs)
            embedding = outputs.logits.mean(dim=1).squeeze().tolist()
           
        # 添加到嵌入結果中
        embedded_data.append({
            "vector": embedding,
            "metadata": chunk["metadata"],
            "text": chunk["text"]  # 添加原始文本內容
        })

    logger.report_text(f"Generated embeddings for {len(chunks)} text chunks using Smollm 360")
    task.close()
    return embedded_data

# 步骤4：存儲嵌入數據到 Qdrant 和 ClearML
#def store_in_qdrant(embedded_data: List[Dict], collection_name: str = "document_embeddings"):
    """將嵌入存儲到 Qdrant 向量數據庫中
    """
    # Determine vector size from the first embedding
    vector_size = len(embedded_data[0]["vector"])

    client = QdrantClient(host="localhost", port=6333)
    task = Task.init(project_name="Featurization_Pipeline", task_name="Store in Qdrant")
    logger = task.get_logger()

    # 每次開始時,刪除集合中的所有向量
    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)
    
    # 創建新集合
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance="Cosine")
    )

    # 將新嵌入數據儲存到集合中
    for idx, data in enumerate(embedded_data):
        client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": str(uuid.uuid4()),  # 使用 UUID 作為唯一標識符
                    "vector": data["vector"],
                    "payload": {
                        "text": data["text"],  # 在 payload 中加入原始文本
                        **data["metadata"]  # 保留元數據
                    }
                }
            ]
        )
        # 添加日志以追蹤每個向量的存儲
        logger.report_text(f"Upserted vector {idx + 1}/{len(embedded_data)} into Qdrant")

    logger.report_text(f"Stored {len(embedded_data)} vectors in Qdrant collection '{collection_name}'")
    task.close()

def store_in_qdrant(embedded_data: List[Dict], collection_name: str = ""):
    """將嵌入存儲到 Qdrant 向量數據庫中"""
    # Determine vector size from the first embedding
    vector_size = len(embedded_data[0]["vector"])

    client = QdrantClient(host="localhost", port=6333)
    task = Task.init(project_name="Featurization_Pipeline", task_name="Store in Qdrant")
    logger = task.get_logger()

    # 如果集合不存在，創建新集合
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance="Cosine")
        )

    # 將嵌入數據插入到集合中
    for idx, data in enumerate(embedded_data):
        client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": str(uuid.uuid4()),  # 使用 UUID 作為唯一標識符
                    "vector": data["vector"],
                    "payload": {
                        "text": data["text"],  # 在 payload 中加入原始文本
                        **data["metadata"]  # 保留元數據
                    }
                }
            ]
        )
        # 添加日志以追蹤每個向量的存儲
        logger.report_text(f"Upserted vector {idx + 1}/{len(embedded_data)} into Qdrant")

    logger.report_text(f"Stored {len(embedded_data)} vectors in Qdrant collection '{collection_name}'")
    task.close()


# 構建完整的 Pipeline
def featurization_pipeline():

    # 初始化MongoDB數據庫和集合名稱
    db_name = "llm_twin"
    collection_name = "Github"
    qdrant_collection_name = "all_embedding"

    # Step 1: Fetch and Decompress Data
    decompressed_data = fetch_and_decompress_data(db_name, collection_name)

    # Step 2: Clean and Chunk Text
    cleaned_chunks = clean_and_chunk_data(decompressed_data)

    # Step 3: Generate Embeddings
    embedded_data = generate_embeddings(cleaned_chunks)

    # Step 4: Store Embeddings in Qdrant and log to ClearML
    store_in_qdrant(embedded_data, qdrant_collection_name)
    print("Featurization pipeline successful")

if __name__ == "__main__":
    featurization_pipeline()
