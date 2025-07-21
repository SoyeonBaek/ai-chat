from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
from typing import Dict
from datetime import datetime
import uvicorn
import asyncio
import os
import httpx
import requests
from rag_search_module import search_relevant_docs, build_prompt
import json
from pydub import AudioSegment
import io
import base64

app = FastAPI()
app.mount("/images", StaticFiles(directory="saved_images"), name="images")
app.mount("/audio", StaticFiles(directory="saved_audio"), name="audio")

origins = ["http://localhost:3000"]
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
        self.active_connections.pop(nickname, None)

    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_json(message)

manager = ConnectionManager()

with open('api-key', 'r') as f:
    API_KEY = f.read().strip()

def convert_webm_base64_to_mp3_base64(base64_audio: str) -> str:
    audio_bytes = base64.b64decode(base64_audio)
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")

    out_buffer = io.BytesIO()
    audio.export(out_buffer, format="mp3")
    out_buffer.seek(0)

    return base64.b64encode(out_buffer.read()).decode("utf-8")

async def chatbot_response(message: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    past_messages = list(messages_collection.find().sort("timestamp", -1).limit(30))[::-1]
    chat_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in past_messages:
        chat_messages.append({"role": msg["role"], "content": msg["message"]})

    data = {"model": "gpt-4", "messages": chat_messages}
    timestamp = datetime.utcnow()
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(url, headers=headers, json=data)
    reply = response.json()["choices"][0]["message"]["content"]

    messages_collection.insert_one({
        "nickname": "chatbot",
        "role": "assistant",
        "message": reply,
        "timestamp": timestamp
    })
    return reply

async def rag_response(query):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    docs = search_relevant_docs(query)
    prompt = build_prompt(docs, query)
    body = {"model": "gpt-3.5-turbo", "messages": prompt}

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(url, headers=headers, json=body)
    reply = response.json()["choices"][0]["message"]["content"]

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
    data = {"model": "dall-e-3", "prompt": prompt, "n": 1, "size": "1024x1024"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=data)
    image_url = response.json()["data"][0]["url"]
    img_data = requests.get(image_url).content
    filename = f"image_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.png"
    filepath = os.path.join("saved_images", filename)
    with open(filepath, "wb") as f:
        f.write(img_data)
    return filename

async def tts_response(text: str) -> str:
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "tts-1",
        "input": text,
        "voice": "nova"  # 선택 가능: alloy, echo, fable, onyx, shimmer
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        audio_bytes = response.content

    filename = f"tts_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.mp3"
    filepath = os.path.join("saved_audio", filename)
    with open(filepath, "wb") as f:
        f.write(audio_bytes)

    return filename

async def stt_response(filename: str, data) -> str:
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }
    filepath = os.path.join("saved_audio", filename)
    base64_audio = data["audio"]
    audio_bytes = base64.b64decode(base64_audio)
    async with httpx.AsyncClient(timeout=30.0) as client:
        with open(filepath, "rb") as audio_file:
            files = {
                "file": (filename, audio_bytes, "audio/webm"),
                "model": (None, "whisper-1"),
            }
            response = await client.post(url, headers=headers, files=files)

    text = response.json()["text"]


    return text

async def talk_response(filename: str, data) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    filepath = os.path.join("saved_audio", filename)

    mp3_base64 = convert_webm_base64_to_mp3_base64(data["audio"])

    past_messages = list(messages_collection.find().sort("timestamp", -1).limit(30))[::-1]
    chat_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in past_messages:
        chat_messages.append({"role": msg["role"], "content": msg["message"]})
    input_audio = { "role": "user", 
      "content": [
        { "type": "text", "text": "내 음성에 대답해줘" },
        { "type": "input_audio", "input_audio": {"data": mp3_base64, "format": "mp3"}}]
    }
    chat_messages.append(input_audio)    

    body = {
      "model": "gpt-4o-audio-preview", 
      "modalities": ["text", "audio"],
      "audio": { "voice": "alloy", "format": "mp3" },
      "messages": chat_messages
    }
    timestamp = datetime.utcnow()
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=body)
    
    if response.status_code != 200:
        print("ERROR STATUS:", response.status_code)
        print("RESPONSE TEXT:", response.text)
        response.raise_for_status()

    msg = response.json()["choices"][0]["message"]

    reply_text = msg.get("text", "")
    audio_base64 = msg.get("audio", {}).get("data", None)

    if audio_base64 is not None:
      # 저장
      filename = f"tts_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.mp3"
      filepath = os.path.join("saved_audio", filename)
      with open(filepath, "wb") as f:
          f.write(base64.b64decode(audio_base64))
      reply_text = response.json()["choices"][0]["message"]["audio"]["transcript"]
      messages_collection.insert_one({
        "nickname": "talk-response",
        "role": "assistant",
        "type": "text",
        "message": reply_text,
        "timestamp": datetime.utcnow()
      })
    else:
      filename = None  # 또는 대체 mp3 반환

      # 텍스트 저장
      messages_collection.insert_one({
        "nickname": "talk-response",
        "role": "assistant",
        "type": "text",
        "message": reply_text,
        "timestamp": datetime.utcnow()
      })


    return filename

async def handle_chatbot(data, nickname):
    query = data.strip()[len("@chatbot"):].strip()
    reply = await chatbot_response(query)
    await manager.broadcast({
        "type": "text",
        "nickname": "chatbot",
        "text": reply,
        "timestamp": datetime.utcnow().isoformat()
    })

async def handle_rag(data, nickname):
    query = data.strip()[len("@rag"):].strip()
    answer = await rag_response(query)
    await manager.broadcast({
        "type": "text",
        "nickname": "rag",
        "text": answer,
        "timestamp": datetime.utcnow().isoformat()
    })

async def handle_image(data, nickname):
    prompt = data.strip()[len("@image"):].strip()
    image = await generate_image(prompt)
    await manager.broadcast({
        "type": "image",
        "nickname": nickname,
        "prompt": prompt,
        "imageUrl": image,
        "timestamp": datetime.utcnow().isoformat()
    })

async def handle_tts(data, nickname):
    text = data.strip()[len("@tts"):].strip()
    filename = await tts_response(text)
    await manager.broadcast({
        "type": "tts",
        "nickname": "tts",
        "content": filename,
        "timestamp": datetime.utcnow().isoformat()
    })

async def handle_stt(filename, data):
    text = await stt_response(filename, data)
    await manager.broadcast({
        "type": "text",
        "nickname": "stt-response",
        "text": text,
        "timestamp": datetime.utcnow().isoformat()
    })

async def handle_talk(filename, data):
    filename = await talk_response(filename, data)
    await manager.broadcast({
        "type": "audio",
        "nickname": "talk-response",
        "content": filename,
        "timestamp": datetime.utcnow().isoformat()
    })

async def broadcast_stt_audio(data, nickname):
    base64_audio = data["audio"]
    filename = f"stt_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.webm"
    filepath = os.path.join("saved_audio", filename)

    with open(filepath, "wb") as f:
        f.write(base64.b64decode(base64_audio))
    
    await manager.broadcast({
        "type": "stt",
        "nickname": data.get("nickname", nickname),
        "content": filename,
        "timestamp": datetime.utcnow().isoformat()
    })
    return filename


@app.websocket("/ws/{nickname}")
async def websocket_endpoint(websocket: WebSocket, nickname: str):
    await manager.connect(websocket, nickname)
    try:
        while True:

            raw_data = await websocket.receive_text()
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                print("Invalid JSON:", raw_data)
                continue

            timestamp = datetime.utcnow()
            data["timestamp"] = timestamp.isoformat()

            messages_collection.insert_one({
                "nickname": data.get("nickname", nickname),
                "type": data.get("type", "text"),
                "role": "user",
                "message": data.get("text", ""),
                "timestamp": timestamp
            })



            msg_type = data.get("type")
            if msg_type == "text":
                await manager.broadcast(data)
                text = data.get("text", "")
                if text.startswith("@chatbot"):
                    asyncio.create_task(handle_chatbot(text, nickname))
                elif text.startswith("@rag"):
                    asyncio.create_task(handle_rag(text, nickname))
                elif text.startswith("@image"):
                    asyncio.create_task(handle_image(text, nickname))
                elif text.startswith("@tts"):
                    asyncio.create_task(handle_tts(text, nickname))
                elif text.strip() == "@stt":
                    # 프론트에서 녹음 UI 표시
                    pass
            elif msg_type == "stt":
                filename = await broadcast_stt_audio(data, nickname)
                asyncio.create_task(handle_stt(filename, data))
            
            elif msg_type == "talk":
                filename = await broadcast_stt_audio(data, nickname)
                asyncio.create_task(handle_stt(filename, data))
                asyncio.create_task(handle_talk(filename, data))


    except WebSocketDisconnect:
        manager.disconnect(nickname)
        await manager.broadcast({
            "type": "system",
            "text": f"{nickname}님이 나갔습니다.",
            "timestamp": datetime.utcnow().isoformat()
        })

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

