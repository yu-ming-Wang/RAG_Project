# Final Project RAG system

Group Member: Yuming Wang, Mingyi Peng

GitHub Name: yu-ming-Wang, Liam M.Y. Peng

HuggingFace Name: domainame, Liam Peng

**Date of Submission:** 2024/12/8

## Instructions on Running the Notebook
To run any notebook associated with this assignment, we run it in our local environment.

---
# Project Report

## Environment and Tooling Milestone
---
1. **Write docker-compose file:** Include Clearml, MongoDB, qdrant, and build our API.
2. **Connect with Clearml:**  Successful!
3. **Connect with Mongodb:** Successful!
4. **Connect with qdrant:** Successful!
5. **Run our API:** Still Adjusting (All our pipelines can run separately in our local host)
6. **Screenshot:** A screenshot showing all docker containers running.

---

## ETL Milestone
1. **Pipeline:** 1. Extract data 2. Transform data 3. Load data (stored in Mongodb)
2. **GitHub Repo Crawler:** We crawl 4 Repository.   (etl_github.py)
3. **Medium Article Crawler:** We crawl 10 Articles. (etl_medium.py)
4. **Youtube Video Crawler:** We crawl 15 videos. (etl_youtube.py)
5. **Links:** All source Links. (links)
6. **Demonstrate:** For Every crawler, we demonstrate one example. Download the notebook file to see the demonstrated result. (etl_demonstrate.ipynb)
---

## Featurization Pipelines Milestone
---
1. **Fetch and decompress data:** Decompress data from MongoDB
2. **Clean and Chunk:** Clean can chunk decompressed text for a later application.
3. **Generate embedding:** Use Smollm 135m tokenizer and model to generate embedding.
4. **Store in qdrant:** Store embedding vector for later similarity search.
5. **Demonstrate:** Use our Medium articles collection from our database for example. (featurization_demonstrate.ipynb)
---

## Finetuning Milestone  
---
1. **First iteration generates answer:** (retrievepipline_smollm135.py, First_iteration_demonstrate.ipynb)
   1. We take the User input to generate embeddings. 
   2. Similarity search from our qdrant collections. 
   3. Generate prompt with user input text and retrieved chunk.
   4. Pass through our LLM Smollm to generate the answer
2. **Finetuning process:** 
3. **Instruct Dataset:** 
4. **Demonstration:** We take the first example question "Tell me how can I navigate to a specific pose - include replanning aspects in your answer." to generate the answer.
---

## Deploying the App Milestone
---
1. **Ollema setup:** 
2. **Webui setup:** 
3. **:** 
4. **:**  
