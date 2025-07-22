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
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from vertexai.preview.generative_models import GenerativeModel
import vertexai
from datetime import datetime, timedelta, timezone
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "applied-might-466615-j0-5ad13d47813b.json"
vertexai.init(project="applied-might-466615-j0", location="us-central1")

# Google Calendar API 인증
SCOPES = ["https://www.googleapis.com/auth/calendar"]

def get_calendar_service():
    creds = None
    if os.path.exists("token.pkl"):
        with open("token.pkl", "rb") as token:
            creds = pickle.load(token)

    if not creds:
        flow = InstalledAppFlow.from_client_secrets_file(
            "client_secret_872350780959-7gg1gbi5iivsq52ohm8geo1jovp4aoao.apps.googleusercontent.com.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.pkl", "wb") as token:
            pickle.dump(creds, token)

    service = build("calendar", "v3", credentials=creds)
    return service


# Gemini 초기화
gemini = GenerativeModel("gemini-2.5-flash")



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

async def parse_schedule_with_gemini(user_input: str):

    prompt = f'''
다음 문장을 읽고, 해야 할 캘린더 작업을 JSON포맷으로 출력해줘 마크다운 넣지 말고..
문장: "{user_input}"
형식:
{{
  "action": "create" | "update" | "delete",
  "title": "지나랑 저녁 약속",
  "datetime": "2024-07-24T19:00:00",
  "new_datetime": "2024-07-24T20:00:00"  // 수정 시만
}}
'''
    response = gemini.generate_content(prompt)
    print("🔍 Gemini 응답 내용:", response.text)
    return response.text

def find_event(service, calendar_id, title, start_time_str):
    # KST 타임존 명시
    
    start_time = datetime.fromisoformat(start_time_str).replace(tzinfo=timezone(timedelta(hours=9)))
    end_time = start_time + timedelta(hours=1)

    # timeMin, timeMax 모두 ISO8601 형식 + 타임존 포함
    events_result = service.events().list(
        calendarId=calendar_id,
        timeMin=start_time.isoformat(),
        timeMax=end_time.isoformat(),
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    for event in events_result.get("items", []):
        if event.get("summary") == title:
            return event
    return None


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


async def handle_calendar(text: str, nickname: str):
    command = text.strip()[len("@calendar"):].strip()
    info = await parse_schedule_with_gemini(command)

    try:
        parsed = json.loads(info)
        action = parsed.get("action", "create")  # 기본은 'create'
        title = parsed["title"]
        start_time = (datetime.fromisoformat(parsed["datetime"])).isoformat()
        end_time = (datetime.fromisoformat(start_time) + timedelta(hours=1)).isoformat()
        new_datetime = parsed.get("new_datetime")
    except Exception as e:
        await manager.broadcast({
            "type": "text",
            "nickname": "calendar",
            "text": f"Gemini가 일정을 파악하지 못했어요: {e}\n{info}",
            "timestamp": datetime.utcnow().isoformat()
        })
        return

    service = get_calendar_service()
    calendar_id = "primary"

    try:
        if action == "create":
            event = {
                "summary": title,
                "start": {"dateTime": start_time, "timeZone": "Asia/Seoul"},
                "end": {"dateTime": end_time, "timeZone": "Asia/Seoul"},
            }
            event_result = service.events().insert(calendarId=calendar_id, body=event).execute()
            response_text = f"✅ '{title}' 일정을 추가했어요! ({start_time})"

        elif action == "delete":
            event = find_event(service, calendar_id, title, start_time)
            if not event:
                raise Exception("일정을 찾을 수 없어요.")
            service.events().delete(calendarId=calendar_id, eventId=event["id"]).execute()
            response_text = f"🗑️ '{title}' 일정을 삭제했어요! ({start_time})"

        elif action == "update":
            if not new_datetime:
                raise Exception("new_datetime 정보가 필요해요.")
            event = find_event(service, calendar_id, title, start_time)
            if not event:
                raise Exception("수정할 일정을 찾을 수 없어요.")
            event["start"]["dateTime"] = (datetime.fromisoformat(new_datetime)).isoformat()
            event["end"]["dateTime"] = (datetime.fromisoformat(new_datetime) + timedelta(hours=1)).isoformat()
            service.events().update(calendarId=calendar_id, eventId=event["id"], body=event).execute()
            response_text = f"🔁 '{title}' 일정을 수정했어요! → {new_datetime}"

        else:
            response_text = f"⚠️ 지원하지 않는 action입니다: {action}"

    except Exception as e:
        response_text = f"❌ 처리 중 오류가 발생했어요: {e}"

    await manager.broadcast({
        "type": "text",
        "nickname": "calendar",
        "text": response_text,
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

async def classify_intent_with_gpt(user_input: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = """
다음 사용자 입력을 읽고 어떤 작업인지 판단해서 하나의 태그로만 알려줘. 반드시 아래 중 하나로만 대답해:

- @chatbot: 일반 대화
- @rag: 문서 기반 질문
- @image: 이미지 생성
- @tts: 텍스트를 음성으로 변환
- @calendar: 일정 관련 (추가, 수정, 삭제)

형식: {"tag": "@calendar"}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    body = {
        "model": "gpt-4",
        "messages": messages
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(url, headers=headers, json=body)
    tag = json.loads(response.json()["choices"][0]["message"]["content"])["tag"]
    return tag

async def handle_ai(text: str, nickname: str):
    command = text.strip()[len("@ai"):].strip()

    try:
        tag = await classify_intent_with_gpt(command)
    except Exception as e:
        await manager.broadcast({
            "type": "text",
            "nickname": "ai",
            "text": f"❌ 태그 분류 실패: {e}",
            "timestamp": datetime.utcnow().isoformat()
        })
        return

    # tag에 따라 적절한 핸들러 호출
    if tag == "@chatbot":
        await handle_chatbot(f"{tag} {command}", nickname)
    elif tag == "@rag":
        await handle_rag(f"{tag} {command}", nickname)
    elif tag == "@image":
        await handle_image(f"{tag} {command}", nickname)
    elif tag == "@tts":
        await handle_tts(f"{tag} {command}", nickname)
    elif tag == "@calendar":
        await handle_calendar(f"{tag} {command}", nickname)
    else:
        await manager.broadcast({
            "type": "text",
            "nickname": "ai",
            "text": f"⚠️ 인식된 태그가 유효하지 않아요: {tag}",
            "timestamp": datetime.utcnow().isoformat()
        })

# function spec 목록
functions = [
    {
        "type": "function",
        "function": {
            "name": "handle_calendar_gpt",
            "description": "사용자의 일정을 추가, 수정, 삭제합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "update", "delete"]
                    },
                    "title": {"type": "string"},
                    "datetime": {"type": "string"},
                    "new_datetime": {"type": "string"}
                },
                "required": ["action", "title", "datetime"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "handle_image_gpt",
            "description": "이미지 생성을 수행합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "handle_tts_gpt",
            "description": "텍스트를 음성으로 변환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "handle_chatbot_gpt",
            "description": "일반 질문에 대해 답변합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"]
            }
        }
    }
]


async def handle_calendar_gpt(action, title, datetime, new_datetime=None):
    fake_input = f"@calendar {title} {datetime}"
    if action == "update":
        fake_input += f" {new_datetime}"
    elif action == "delete":
        fake_input += " 취소"
    await handle_calendar(fake_input, nickname="gpt")

async def handle_image_gpt(prompt):
    image = await generate_image(prompt)
    await manager.broadcast({
        "type": "image",
        "nickname": "gpt",
        "prompt": prompt,
        "imageUrl": image,
        "timestamp": datetime.utcnow().isoformat()
    })

async def handle_tts_gpt(text):
    filename = await tts_response(text)
    await manager.broadcast({
        "type": "tts",
        "nickname": "gpt",
        "content": filename,
        "timestamp": datetime.utcnow().isoformat()
    })

async def handle_chatbot_gpt(question):
    reply = await chatbot_response(question)
    await manager.broadcast({
        "type": "text",
        "nickname": "gpt",
        "text": reply,
        "timestamp": datetime.utcnow().isoformat()
    })


from openai import OpenAI

openai_client = OpenAI(api_key=API_KEY)

async def handle_gpt_function(text: str):
    user_input = text.strip()[len("@gpt"):].strip()

    response = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": user_input}],
        tools=functions,
        tool_choice="auto"
    )

    choice = response.choices[0]
    if hasattr(choice, "message") and choice.message.tool_calls:
        for call in choice.message.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)

            # 이름 기반 함수 실행
            if name == "handle_calendar_gpt":
                await handle_calendar_gpt(**args)
            elif name == "handle_image_gpt":
                await handle_image_gpt(**args)
            elif name == "handle_tts_gpt":
                await handle_tts_gpt(**args)
            elif name == "handle_chatbot_gpt":
                await handle_chatbot_gpt(**args)



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
                elif text.startswith("@calendar"):
                    asyncio.create_task(handle_calendar(text, nickname))
                elif text.startswith("@ai"):
                    asyncio.create_task(handle_ai(text, nickname))
                elif text.startswith("@gpt"):
                    asyncio.create_task(handle_gpt_function(text))
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

