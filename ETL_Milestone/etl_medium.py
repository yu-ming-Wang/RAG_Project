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
from selenium.webdriver.edge.options import Options
import pickle
import logging

class MediumCrawlerEdgeWithCookies:
    def __init__(self, cookies_file="cookies.pkl"):
        # 初始化瀏覽器選項
        options = Options()
        options.add_argument("--headless")  # 無界面模式
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36")  # 模擬正常瀏覽器
        self.driver = webdriver.Edge(options=options)
        self.scroll_limit = 5  # 頁面滾動次數
        self.cookies_file = cookies_file  # 存放 Cookies 的檔案路徑

    def save_cookies(self):
        """保存 Cookies 到檔案"""
        with open(self.cookies_file, "wb") as f:
            pickle.dump(self.driver.get_cookies(), f)
        logging.info("Cookies 已保存。")

    def load_cookies(self):
        """從檔案加載 Cookies"""
        try:
            with open(self.cookies_file, "rb") as f:
                cookies = pickle.load(f)
            for cookie in cookies:
                self.driver.add_cookie(cookie)
            logging.info("Cookies 已加載。")
        except FileNotFoundError:
            logging.warning("未找到 Cookies 檔案，將從頭開始爬取。")

    def _scroll_page(self):
        """滾動頁面以加載更多內容"""
        current_scroll = 0
        while current_scroll < self.scroll_limit:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # 等待頁面加載
            current_scroll += 1

    def extract(self, link: str) -> dict:
        """從 Medium 網址提取文章內容"""
        try:
            logging.info(f"訪問鏈接: {link}")
            self.driver.get(link)

            # 加載 Cookies
            self.load_cookies()
            self.driver.refresh()  # 使用 Cookies 後刷新頁面
            time.sleep(2)

            # 滾動頁面確保內容完全加載
            self._scroll_page()

            # 解析頁面內容
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

            # 保存 Cookies
            self.save_cookies()
            return article_data
        except Exception as e:
            logging.error(f"提取 Medium 文章失敗: {e}")
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

def extract_data(user_full_name: str, links: List[str], project_name: str = "Test_medium_ETL") -> Dict:
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
    task = Task.init(project_name=project_name, task_name="Extract medium Data")

    # Dictionary to store crawled data
    crawled_data = {}

    # Extract data from each link
    for link in links:
        if "medium.com" in link:
            crawled_data[link] = MediumCrawlerEdgeWithCookies().extract(link)
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

#轉成githubCraw下來的形式
def transform_medium_to_github_format(data):
    transformed_data = {}
    for url, entry in data.items():
        # 初始化內容為空字典
        content_parts = {}
        
        # 取得 Content, Subtitle, Title
        content = entry.get("Content", "")
        subtitle = entry.get("Subtitle", "")
        title = entry.get("Title", "")
        
        # 將標題和副標題作為單獨的部分
        if title:
            content_parts["part1"] = f"Title: {title}"
        if subtitle:
            content_parts["part2"] = f"Subtitle: {subtitle}"
        
        # 將其餘內容作為另一部分
        content_parts["part3"] = content

        # 將結果儲存到轉換後的資料中
        transformed_data[url] = {
            "content": content_parts
        }

    return transformed_data


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
    task = Task.init(project_name="Test_medium_ETL", task_name="Transform medium Data")
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
    task = Task.init(project_name="Test_medium_ETL", task_name="Load medium Data")
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
    logging.basicConfig(level=logging.INFO)
    
    user_full_name = "Yu-Ming Wang"
    links = [   "https://medium.com/schmiedeone/getting-started-with-ros2-part-1-d4c3b7335c71",
                "https://medium.com/@santoshbalaji/manipulation-with-moveit2-visualizing-robot-arm-in-simulation-1-8cd3a46d42b4",
                "https://medium.com/@CanyonLakeRobotics/tuning-the-ros2-nav2-stack-5b01f455e217",
                "https://medium.com/creating-a-gazebo-simulation-with-ros2-for-your/introduction-8daf6efa12f4",
                "https://medium.com/@arshad.mehmood/a-guide-to-multi-robot-navigation-utilizing-turtlebot3-and-nav2-cd24f96d19c6",
                "https://medium.com/@arshad.mehmood/ros2-gazebo-world-map-generator-a103b510a7e5",
                "https://medium.com/@kabilankb2003/exploring-nvidia-carter-robot-in-ros2-navigation-with-isaac-sim-and-action-graph-8186a1398361",
                "https://medium.com/geekculture/reinforcement-learning-path-planner-for-6dof-robot-in-ros2-518581dc72c7",
                "https://medium.com/@thehummingbird/building-a-ros2-project-part-1-a2c02d6ac3d8",
                "https://medium.com/schmiedeone/getting-started-with-ros2-part-1-d4c3b7335c71"
]
    
    # Extract data using the function
    extracted_data = extract_data(user_full_name, links)
   
    extracted_data = transform_medium_to_github_format (extracted_data)

    # Step 2: Transform Data
    transformed_data = transform_data(extracted_data)

    # step 3: Load data
    load_data(transformed_data, db_name="llm_twin", collection_name="test_medium")

    print("Medium atticle extract、transform and insert success！")

