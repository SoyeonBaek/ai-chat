import React, { useState, useEffect, useRef } from "react";
import "./App.css";  // CSS파일 import

function App() {
  const [nickname, setNickname] = useState("");
  const [inputNickname, setInputNickname] = useState("");
  const [message, setMessage] = useState("");
  const [chatLog, setChatLog] = useState([]);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [showRecorder, setShowRecorder] = useState(false);
  const ws = useRef(null);
  const messagesEndRef = useRef(null);
  const recordingIntervalRef = useRef(null);

  useEffect(() => {
    if (!nickname) return;

    ws.current = new WebSocket(`ws://localhost:8000/ws/${nickname}`);

    ws.current.onopen = () => {
      console.log("WebSocket connected");
    };

    ws.current.onmessage = (event) => {
      try {
        const msgObj = JSON.parse(event.data);
        setChatLog((prev) => [...prev, msgObj]);

        if (msgObj.nickname === nickname && (msgObj.text === "@stt" || msgObj.text == "@talk")) {
          setShowRecorder(true);
        }
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

  const sendMessage = () => {
    if (message.trim() === "") return;

    const jsonPayload = {
      type: "text",
      nickname: nickname,
      text: message,
      timestamp: new Date().toISOString()
    };

    ws.current.send(JSON.stringify(jsonPayload));
    setMessage("");

  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatLog]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    let chunks = [];

    recorder.ondataavailable = (e) => {
      chunks.push(e.data);
    };

    recorder.onstop = () => {
      setRecordedChunks(chunks);
      clearInterval(recordingIntervalRef.current);
    };

    recorder.start();
    setMediaRecorder(recorder);
    setIsRecording(true);
    setRecordingTime(0);

    recordingIntervalRef.current = setInterval(() => {
      setRecordingTime((prev) => prev + 1);
    }, 1000);
  };

  const stopRecording = () => {
    mediaRecorder?.stop();
    setIsRecording(false);
  };

  const sendToSTTOverWS = async () => {
    if (!recordedChunks.length || !ws.current) return;
    const blob = new Blob(recordedChunks, { type: "audio/webm" });
    const reader = new FileReader();

    reader.onloadend = () => {
      const base64Audio = reader.result.split(",")[1];
      ws.current.send(JSON.stringify({ type: "stt", audio: base64Audio }));
      setShowRecorder(false);
    };

    reader.readAsDataURL(blob);
  };
  const sendToTALKOverWS = async () => {
    if (!recordedChunks.length || !ws.current) return;
    const blob = new Blob(recordedChunks, { type: "audio/webm" });
    const reader = new FileReader();

    reader.onloadend = () => {
      const base64Audio = reader.result.split(",")[1];
      ws.current.send(JSON.stringify({ type: "talk", audio: base64Audio }));
      setShowRecorder(false);
    };

    reader.readAsDataURL(blob);
  };

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

  const renderMessage = (msgObj, i) => {
    if (msgObj.type === "image") {
      const url = `http://localhost:8000/images/${msgObj.imageUrl}`;
      return (
        <div key={i} className="message">
          <strong>{msgObj.nickname}</strong> [{msgObj.timestamp}]<br />
          <img src={url} alt="Generated" style={{ maxWidth: "300px", borderRadius: "8px" }} />
        </div>
      );
    }

    if (msgObj.type === "text") {
      return (
        <div key={i} className="message">
          <strong>{msgObj.nickname}</strong> [{msgObj.timestamp}] : {msgObj.text}
        </div>
      );
    }
    if (msgObj.type === "tts" || msgObj.type === "stt" | msgObj.type === "audio") {
      const url = `http://localhost:8000/audio/${msgObj.content}`;
      return (
        <div key={i} className="message">
          <audio controls src={url} style={{ marginTop: "10px" }} />
        </div>
      );
    }

    return null;
  };

  return (
    <div className="chat-container">
      <h2>채팅방 (닉네임: {nickname})</h2>
      <div className="chat-messages">
        {chatLog.map((msg, i) => renderMessage(msg, i))}
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

      {showRecorder && (
        <div className="recorder-bar">
          <p>녹음 중: {recordingTime}초</p>
          <button onClick={startRecording} disabled={isRecording}>녹음 시작</button>
          <button onClick={stopRecording} disabled={!isRecording}>녹음 끝</button>
          <button onClick={sendToSTTOverWS} disabled={!recordedChunks.length}>STT 전송</button>
          <button onClick={sendToTALKOverWS} disabled={!recordedChunks.length}>TALK 전송</button>
        </div>
      )}
    </div>
  );
}

export default App;
