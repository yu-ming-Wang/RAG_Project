from pymongo import MongoClient

def test_mongodb_connection():
    # 替換為正確的 MongoDB 連接 URL
    client = MongoClient("mongodb://localhost:27017/")
    db = client["etl_pipeline"]
    collection = db["media_sources"]

    # 插入測試數據
    test_data = {"test": "MongoDB connection successful!"}
    collection.insert_one(test_data)

    # 驗證數據是否插入成功
    result = collection.find_one({"test": "MongoDB connection successful!"})
    assert result is not None, "Test failed: MongoDB data not inserted properly"
    print("Test passed: MongoDB connection and data insertion successful!")

if __name__ == "__main__":
    test_mongodb_connection()
