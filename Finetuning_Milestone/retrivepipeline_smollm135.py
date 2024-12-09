from typing import List, Dict
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from clearml import Task

huggingface_token = "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"
model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=huggingface_token).half()
# Add padding token to avoid padding error
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def retrieve_relevant_chunks(user_input: str, collection_name: str = "", top_k: int = 5) -> List[Dict]:
    """根据用户输入从 Qdrant 中检索相关的文本块"""
    task = Task.init(project_name="Retrive_Pipeline", task_name="User Input Retrieval")
    logger = task.get_logger()

    client = QdrantClient(host="localhost", port=6333)

    print(f"Using device: {device}")

    try:
        logger.report_text(f"Generating embedding for user input: {user_input}")
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        print("Tokenized inputs:", inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            user_embedding = outputs.logits.mean(dim=1).squeeze().tolist()
            print("Generated embedding size:", len(user_embedding))
            print("Generated embedding:", user_embedding[:3])  # 打印嵌入前5个值

        logger.report_text("User input embedding generated successfully.")
    except Exception as e:
        logger.report_text(f"Error generating embedding for user input: {e}")
        print(f"Error generating embedding: {e}")
        task.close()
        return []

    try:
        logger.report_text("Retrieving relevant chunks from Qdrant.")
        collections = client.get_collections()
        print("Available collections in Qdrant:", collections)

        search_result = client.search(
            collection_name=collection_name,
            query_vector=user_embedding,
            limit=top_k
        )
        print("Search results:", search_result)

        logger.report_text(f"Retrieved {len(search_result)} relevant chunks from Qdrant.")
    except Exception as e:
        logger.report_text(f"Error retrieving chunks from Qdrant: {e}")
        print(f"Error retrieving chunks: {e}")
        task.close()
        return []

    retrieved_chunks = []
    for point in search_result:
        if "text" in point.payload:
            retrieved_chunks.append({"text": point.payload["text"], "score": point.score})
        else:
            print("Warning: 'text' key not found in payload.")
            logger.report_text("Warning: 'text' key not found in payload.")

    task.close()
    return retrieved_chunks

def generate_prompt(user_input: str, retrieved_chunks: List[Dict]) -> str:
    """生成包含检索到的文档内容和用户输入的 Prompt"""
    prompt_template = "Here are some relevant documents to answer your query:\n"
    for chunk in retrieved_chunks[:4]:  # 限制为前3个片段
        prompt_template += f"- {chunk['text']}\n"

    prompt_template += f"\nUser question: {user_input}\nAnswer:"
    print("Generated prompt:", prompt_template[:500])  # 打印前500字符的 prompt
    return prompt_template

def generate_answer(prompt: str) -> str:
    """使用 smollm 135 生成回答"""
    task = Task.init(project_name="Retrive_Pipeline", task_name="Response Generation")
    logger = task.get_logger()

    try:
        logger.report_text("Generating answer using smollm 135.")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        print("Prompt inputs:", inputs)

        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],  # 确保 attention_mask 被传递
                max_new_tokens=150,  # 限制生成的 token 数量
                do_sample=True,  # 启用采样模式
                temperature=0.7  # 控制生成多样性
            )
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated response:", response_text)
        logger.report_text("Answer generated successfully.")
    except Exception as e:
        logger.report_text(f"Error generating answer: {e}")
        print(f"Error generating answer: {e}")
        response_text = "Sorry, I couldn't generate an answer at this time."

    task.close()
    return response_text

if __name__ == "__main__":
    user_input = "Tell me how can I navigate to a specific pose - include replanning aspects in your answer."
    #user_input_2 = "Can you provide me with code for this task?"

    retrieved_chunks = retrieve_relevant_chunks(user_input, "github_embedding", 5)
    if not retrieved_chunks:
        print("No relevant documents found.")
    else:
        prompt = generate_prompt(user_input, retrieved_chunks)
        answer = generate_answer(prompt)
        print("Generated Answer:\n", answer)