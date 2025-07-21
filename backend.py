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
import time
from fastapi.staticfiles import StaticFiles
import os 
import asyncio
import httpx

app = FastAPI()
app.mount("/images", StaticFiles(directory="saved_images"), name="images")

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
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)

    #response = requests.post(url, headers=headers, json=data)
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
    url = "https://api.openai.com/v1/chat/completions"
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
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(url, headers=headers, json=body)
        reply = response.json()["choices"][0]["message"]["content"]
    # chatbot 메시지도 DB 저장
        messages_collection.insert_one({
          "nickname": "rag",
          "role": "assistant",
          "message": reply,
          "timestamp": datetime.utcnow()
        })

        return reply


async def generate_image(prompt: str) -> str:
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "dall-e-3",  # 또는 "dall-e-2"
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=data)
        #response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        image_url = response.json()["data"][0]["url"]
        img_data = requests.get(image_url).content
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"image_{timestamp}.png"
        filepath = os.path.join("saved_images", filename)
        with open(filepath, "wb") as f:
            f.write(img_data)
    
        return filename


async def handle_chatbot(data, nickname):
    bot_query = data.strip()[len("@chatbot"):].strip()
    bot_reply = await chatbot_response(bot_query)
    timestamp = datetime.utcnow()
    bot_message = f"chatbot [{timestamp}] : {bot_reply}"
    await manager.broadcast(bot_message)

async def handle_rag(data, nickname):
    query = data.strip()[len("@rag"):].strip()
    answer = await rag_response(query)
    timestamp = datetime.utcnow()
    rag_message = f"rag [{timestamp}] : {answer}"
    await manager.broadcast(rag_message)

async def handle_image(data, nickname):
    image_prompt = data.strip()[len("@image"):].strip()
    image = await generate_image(image_prompt)
    await manager.broadcast(f"[IMAGE]: {image}")


@app.websocket("/ws/{nickname}")
async def websocket_endpoint(websocket: WebSocket, nickname: str):
    await manager.connect(websocket, nickname)
    try:
        while True:
            data = await websocket.receive_text()
            timestamp = datetime.utcnow()  # UTC 기준 현재 시간
            full_message = f'{nickname} [{timestamp.strftime("%H:%M")}] : {data}'
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
                asyncio.create_task(handle_chatbot(data, nickname))

            elif data.strip().startswith("@rag"):
                asyncio.create_task(handle_rag(data, nickname))

            elif data.strip().startswith("@image"):
                asyncio.create_task(handle_image(data, nickname))
          

    
    except WebSocketDisconnect:
        manager.disconnect(nickname)
        await manager.broadcast(f"{nickname}님이 나갔습니다.")

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

