import requests
import numpy as np
from pymongo import MongoClient

with open('api-key', 'r') as f:
    API_KEY = f.read().strip()
EMBEDDING_MODEL = "text-embedding-3-small"

client = MongoClient("mongodb://localhost:27017")
collection = client.rag.documents


# 코사인 유사도
def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 이미 저장된 임베딩 기반으로 유사 문서 검색
def search_relevant_docs(query: str, top_k=3):
    q_emb = embed_text(query)
    docs = []
    for doc in collection.find():
        sim = cosine_sim(q_emb, doc["embedding"])
        docs.append((sim, doc["text"]))
    docs.sort(reverse=True)
    return [d[1] for d in docs[:top_k]]

# 질문 임베딩
def embed_text(text: str):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}" ,
        "Content-Type": "application/json"
    }
    data = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["data"][0]["embedding"]

# GPT 프롬프트 구성
def build_prompt(context_docs, user_question):
    context = "\n---\n".join(context_docs)
    return [
        {"role": "system", "content": "You are a helpful assistant using provided documents."},
        {"role": "user", "content": f"Refer to the documents below and answer the question.\n\nDocuments:\n{context}\n\nQuestion: {user_question}"}
    ]
