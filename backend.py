from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json

app = FastAPI()

# CORS 설정 (개발 중 React 등에서 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 연결된 클라이언트 목록
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        text = json.dumps(message)
        for connection in self.active_connections:
            await connection.send_text(text)

manager = ConnectionManager()

@app.websocket("/ws/{nickname}")
async def websocket_endpoint(websocket: WebSocket, nickname: str):
    await manager.connect(websocket)
    try:
        while True:
            text_data = await websocket.receive_text()
            message = {
                "sender": nickname,
                "message": text_data,
                "timestamp": None  # 원하면 datetime.utcnow().isoformat() 등 추가 가능
            }
            await manager.broadcast(message)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

