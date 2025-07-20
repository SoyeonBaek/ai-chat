# backend.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from typing import List, Dict
import uvicorn
from datetime import datetime

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
                "message": data,
                "timestamp": timestamp
            })
            # 브로드캐스트
            await manager.broadcast(full_message)
    except WebSocketDisconnect:
        manager.disconnect(nickname)
        await manager.broadcast(f"{nickname}님이 나갔습니다.")

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

