# backend.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from typing import List, Dict
import uvicorn
from datetime import datetime
import requests
import openai
from rag_search_module import search_relevant_docs, build_prompt

app = FastAPI()

# CORS 허용 (React 개발 서버 주소 넣기)
origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient("mongodb://localhost:27017")
db = client.chat_db
messages_collection = db.messages

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, nickname: str):
        await websocket.accept()
        self.active_connections[nickname] = websocket

    def disconnect(self, nickname: str):
        if nickname in self.active_connections:
            del self.active_connections[nickname]

    async def send_personal_message(self, message: str, nickname: str):
        websocket = self.active_connections.get(nickname)
        if websocket:
            await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

with open('api-key', 'r') as f:
    API_KEY = f.read().strip()

# @chatbot 처리 함수
async def chatbot_response(message: str) -> str:

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
      "Authorization": f"Bearer {API_KEY}",
      "Content-Type": "application/json",
    }
      
    # 이전 대화 불러오기(이전 30개)
    past_messages = list(messages_collection.find().sort("timestamp", -1).limit(30))[::-1]

    # GPT에게 보낼 messages 생성
    chat_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in past_messages:
        chat_messages.append({"role": msg["role"], "content": msg["message"]})

    data = {
      "model": "gpt-4",
      "messages":chat_messages
    }

    timestamp = datetime.utcnow()  # UTC 기준 현재 시간
    response = requests.post(url, headers=headers, json=data)
    reply = response.json()["choices"][0]["message"]["content"]
    
    # chatbot 메시지도 DB 저장
    messages_collection.insert_one({
      "nickname": "chatbot",
      "role": "assistant",
      "message": reply,
      "timestamp": timestamp
    })
    
    return reply  


# @rag 처리 함수
async def rag_response(query):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    docs = search_relevant_docs(query)
    prompt = build_prompt(docs, query)
    
    body = {
        "model": "gpt-3.5-turbo",
        "messages": prompt
    }
    
    res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
    reply = res.json()["choices"][0]["message"]["content"]
    # chatbot 메시지도 DB 저장
    messages_collection.insert_one({
      "nickname": "rag",
      "role": "assistant",
      "message": reply,
      "timestamp": datetime.utcnow()
    })

    return reply

@app.websocket("/ws/{nickname}")
async def websocket_endpoint(websocket: WebSocket, nickname: str):
    await manager.connect(websocket, nickname)
    try:
        while True:
            data = await websocket.receive_text()
            timestamp = datetime.utcnow()  # UTC 기준 현재 시간
            full_message = f'{nickname} [{timestamp.strftime("%Y-%m-%d %H:%M:%S")}] : {data}'
            # DB 저장 (시간도 같이)
            messages_collection.insert_one({
                "nickname": nickname,
                "role": "user",
                "message": data,
                "timestamp": timestamp
            })
            # 브로드캐스트
            await manager.broadcast(full_message)

            if data.strip().startswith("@chatbot"):
                bot_query = data.strip()[len("@chatbot"):].strip()
                bot_reply = await chatbot_response(bot_query)
                bot_message = f"chatbot [{timestamp}] : {bot_reply}"
                await manager.broadcast(bot_message)
            
            if data.strip().startswith("@rag"):
                query = data.strip()[len("@rag"):].strip()
                answer = await rag_response(query)
                rag_message = f"chatbot [{timestamp}] : {answer}"
                await manager.broadcast(rag_message)

    except WebSocketDisconnect:
        manager.disconnect(nickname)
        await manager.broadcast(f"{nickname}님이 나갔습니다.")

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

