import urllib
from clearml import Task
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import logging
import os
import shutil
import subprocess
import tempfile
import stat
import json
from typing import Dict

class MediumCrawler:
    def __init__(self):
        options = Options()
        options.add_argument("--headless")  # 无界面模式
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        self.driver = webdriver.Chrome(options=options)
        self.scroll_limit = 5  # 控制页面滚动次数

    def _scroll_page(self):
        """滚动页面以加载完整内容"""
        current_scroll = 0
        while current_scroll < self.scroll_limit:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # 等待页面加载
            current_scroll += 1

    def extract(self, link: str) -> dict:
        """从 Medium 链接中提取文章内容"""
        try:
            logging.info(f"访问链接: {link}")
            self.driver.get(link)
            self._scroll_page()  # 确保页面完全加载

            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            title = soup.find("h1").get_text(strip=True) if soup.find("h1") else None
            subtitle = soup.find("h2").get_text(strip=True) if soup.find("h2") else None
            content = "\n".join([p.get_text(strip=True) for p in soup.find_all("p")])

            article_data = {
                "Title": title,
                "Subtitle": subtitle,
                "Content": content,
                "Source": "Medium",
                "Link": link,
            }
            logging.info(f"成功提取文章: {title}")
            return article_data
        except Exception as e:
            logging.error(f"提取 Medium 文章失败: {e}")
            return {}
        finally:
            self.driver.quit()

class GithubCrawler:
    def __init__(self, ignore_patterns=None):
        """初始化忽略文件或目录的模式"""
        self.ignore_patterns = ignore_patterns or [".git", ".toml", ".lock", ".png"]

    def _is_ignored(self, file_path: str) -> bool:
        """判断文件是否需要被忽略"""
        for pattern in self.ignore_patterns:
            if file_path.endswith(pattern) or pattern in file_path:
                return True
        return False

    def extract(self, repo_url: str) -> dict:
        """从 GitHub 仓库提取文件内容"""
        def handle_remove_readonly(func, path, exc_info):
            """处理无法删除的文件，强制解除只读属性并重试"""
            os.chmod(path, stat.S_IWRITE)
            func(path)

        temp_dir = None
        try:
            logging.info(f"克隆仓库: {repo_url}")
            temp_dir = tempfile.mkdtemp()  # 创建临时目录
            subprocess.run(["git", "clone", repo_url, temp_dir], check=True)

            repo_data = {}
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, temp_dir)
                    if self._is_ignored(relative_path):
                        continue
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            repo_data[relative_path] = f.read()
                    except Exception as e:
                        logging.warning(f"读取文件失败: {file_path}, 错误: {e}")

            logging.info(f"成功提取仓库数据: {repo_url}")
            return {"repository": repo_url, "content": repo_data}
        except Exception as e:
            logging.error(f"提取 GitHub 仓库失败: {e}")
            return {}
        finally:
            if temp_dir:  # 确保临时目录存在
                try:
                    shutil.rmtree(temp_dir, onexc=handle_remove_readonly)
                except Exception as e:
                    logging.error(f"删除临时目录时出错: {e}")
from clearml import Task
from typing import Dict, List

def extract_data(user_full_name: str, links: List[str], project_name: str = "Test_github_ETL") -> Dict:
    """
    Extracts data from Medium and GitHub links, logs and uploads the data to ClearML.

    Args:
    user_full_name (str): The user's full name.
    links (List[str]): A list of links to extract data from.
    project_name (str): The ClearML project name. Default is "Test_ETL".

    Returns:
    Dict: Extracted data from the provided links.
    """
    # Initialize ClearML Task
    task = Task.init(project_name=project_name, task_name="Extract github Data")

    # Dictionary to store crawled data
    crawled_data = {}

    # Extract data from each link
    for link in links:
        if "medium.com" in link:
            crawled_data[link] = MediumCrawler().extract(link)
        elif "github.com" in link:
            crawled_data[link] = GithubCrawler().extract(link)

    # Log crawled data to ClearML
    logger = task.get_logger()
    logger.report_text(f"Crawled data: {crawled_data}")

    # Upload raw data as an artifact for future usage
    task.upload_artifact("raw_data", crawled_data)

    # Close the ClearML Task
    task.close()

    return crawled_data

# Step 2: Transform Data
def extract_source_metadata(link: str) -> Dict:
    """从链接中提取元数据"""
    parsed_url = urllib.parse.urlparse(link)
    metadata = {
        "domain": parsed_url.netloc,
        "path": parsed_url.path,
        "query": parsed_url.query,
    }
    return metadata

def clean_and_standardize(content: str) -> str:
    """清理并标准化内容"""
    # 在这里加入清理和标准化的逻辑，例如去除多余的空格，转换为统一格式等
    return content

import json
import gzip
import base64

def compress_content(content: str) -> str:
    """压缩内容并转换为base64字符串"""
    compressed_bytes = gzip.compress(content.encode('utf-8'))
    return base64.b64encode(compressed_bytes).decode('utf-8')

def transform_data(raw_data: Dict) -> Dict:
    task = Task.init(project_name="Test_github_ETL", task_name="Transform github Data")
    transformed_data = {}
    
    for link, content_dict in raw_data.items():
        content_data = content_dict.get("content", {})
        
        if isinstance(content_data, dict):
            transformed_content = {}
            for file_name, file_content in content_data.items():
                # 压缩大文件内容
                transformed_content[file_name] = {
                    "content": compress_content(file_content),
                    "type": "compressed"  # 标记这个文件内容是压缩的
                }
        
            transformed_data[link] = {
                "content": transformed_content,
                "source": extract_source_metadata(link),
            }
        else:
            transformed_data[link] = {
                "content": {
                    "content": compress_content(content_data),
                    "type": "compressed"  # 标记这个内容是压缩的
                },
                "source": extract_source_metadata(link),
            }
    
    # 将转换后的数据作为工件上传到 ClearML
    task.upload_artifact("transformed_data", transformed_data)
    task.close()
    return transformed_data

from pymongo import MongoClient
from gridfs import GridFS

# Step 3: Load Data to MongoDB
def load_data(transformed_data: Dict, db_name: str = "llm_twin", collection_name: str = ""):
    task = Task.init(project_name="Test_github_ETL", task_name="Load github Data")
    logger = task.get_logger()
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client[db_name]
    collection = db[collection_name]
    fs = GridFS(db)  # 初始化 GridFS

    logger.report_text("成功連接到 MongoDB！")

    for doc in transformed_data.values():
        # 計算文檔大小
        doc_size = len(json.dumps(doc).encode('utf-8'))

        if doc_size > 16 * 1024 * 1024:  # 超過 MongoDB 文檔大小限制
            logger.report_text(f"文檔過大，將使用 GridFS 存儲，大小: {doc_size} bytes")
            
            # 存儲超大文件到 GridFS
            gridfs_id = fs.put(json.dumps(doc).encode('utf-8'))
            
            # 在主集合中存儲引用
            doc_reference = {
                "gridfs_id": gridfs_id,
                "metadata": {key: doc[key] for key in doc if key != "content"}  # 儲存內容以外的元數據
            }
            insert_result = collection.insert_one(doc_reference)
            logger.report_text(f"使用 GridFS 插入的文檔引用 ID: {insert_result.inserted_id}")
        else:
            # 正常插入
            insert_result = collection.insert_one(doc)
            logger.report_text(f"插入的文檔 ID: {insert_result.inserted_id}")

    task.close()



# 使用 ClearML 跟踪爬取任务
if __name__ == "__main__":
  
    user_full_name = "Yu-Ming Wang"
    links = ["https://github.com/moveit/moveit2",
             "https://github.com/ros-infrastructure/www.ros.org",
             "https://github.com/ros-navigation/docs.nav2.org",
             "https://github.com/gazebosim/docs"
    ]
    
    # Extract data using the function
    extracted_data = extract_data(user_full_name, links)
   
    # Step 2: Transform Data
    transformed_data = transform_data(extracted_data)

    # step 3: Load data
    load_data(transformed_data, db_name="llm_twin", collection_name="test_github")

    print("Github extract、transform and insert success！")


