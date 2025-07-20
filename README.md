# frontend 실행

cd chat-frontend  
npm start

# backend 실행

uvicorn backend:app --host 0.0.0.0 --port 8000

# mongodb 실행

brew services start mongodb-community@6.0

---

1. chatbot

@chatbot <Your question>

- 챗봇은 당신의 최근 30개의 채팅 이력을 함께 제공합니다.

---

2. rag

@rag <Your question>

- 저장되어 있는 문서들의 검색을 포함합니다.  
- 문서의 임베딩은 Python 스크립트 embed_documents.py를 통해 수행하세요.


