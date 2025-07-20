import React, { useState, useEffect, useRef } from "react";
import "./App.css";  // CSS파일 import (아래 스타일 참고)

function App() {
  const [nickname, setNickname] = useState("");
  const [inputNickname, setInputNickname] = useState("");
  const [message, setMessage] = useState("");
  const [chatLog, setChatLog] = useState([]);
  const ws = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (!nickname) return;

    ws.current = new WebSocket(`ws://localhost:8000/ws/${nickname}`);

    ws.current.onopen = () => {
      console.log("WebSocket connected");
    };

    ws.current.onmessage = (event) => {
      try {
        // 백엔드가 그냥 텍스트 메시지로 보내므로 JSON 파싱은 안함
        const msg = event.data;
        setChatLog((prev) => [...prev, msg]);
      } catch (e) {
        console.error("Invalid message format:", event.data);
      }
    };

    ws.current.onclose = () => {
      console.log("WebSocket disconnected");
    };

    return () => {
      ws.current.close();
    };
  }, [nickname]);

  // 메시지 전송 함수
  const sendMessage = () => {
    if (message.trim() === "") return;
    ws.current.send(message);
    setMessage("");
  };

  // 채팅창 스크롤 자동 아래로
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatLog]);

  // 닉네임 설정 UI
// 닉네임 입력 화면 부분 수정
  if (!nickname) {
    return (
      <div className="nickname-container">
        <h2>닉네임을 입력하세요</h2>
        <input
          type="text"
          value={inputNickname}
          onChange={(e) => setInputNickname(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && inputNickname.trim() !== "") {
              setNickname(inputNickname.trim());
            }
          }}
          placeholder="닉네임"
        />
        <button
          onClick={() => {
            if (inputNickname.trim() !== "") setNickname(inputNickname.trim());
          }}
        >
          입장
        </button>
      </div>
    );
  } 

  // 채팅 UI
  return (
    <div className="chat-container">
      <h2>채팅방 (닉네임: {nickname})</h2>
      <div className="chat-messages">
        {chatLog.map((msg, i) => (
          <div key={i} className="message">
            {/* 메시지가 문자열이라서, 닉네임 부분만 스타일링 하려면 backend 형식 변경 필요 */}
            {msg}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="input-area">
        <input
          type="text"
          placeholder="메시지를 입력하세요"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => {
            if (e.nativeEvent.isComposing) return;
            if (e.key === "Enter") sendMessage();
          }}
        />
        <button onClick={sendMessage}>전송</button>
      </div>
    </div>
  );
}

export default App;

