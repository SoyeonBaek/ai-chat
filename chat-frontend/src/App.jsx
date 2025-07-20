import React, { useState, useEffect, useRef } from "react";

function App() {
  const [nickname, setNickname] = useState("");
  const [inputNickname, setInputNickname] = useState("");
  const [message, setMessage] = useState("");
  const [chatLog, setChatLog] = useState([]);
  const ws = useRef(null);

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
      } catch (e) {
        console.error("Invalid JSON:", event.data);
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


    // 서버로 전송
    ws.current.send(message);
    setMessage("");
  };

  if (!nickname) {
    return (
      <div style={{ padding: 20 }}>
        <h2>Set your nickname</h2>
        <input
          type="text"
          value={inputNickname}
          onChange={(e) => setInputNickname(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && inputNickname.trim() !== "") {
              setNickname(inputNickname.trim());
            }
          }}
        />
        <button
          onClick={() => {
            if (inputNickname.trim() !== "") setNickname(inputNickname.trim());
          }}
        >
          Enter
        </button>
      </div>
    );
  }

  return (
    <div style={{ padding: 20 }}>
      <h2>Chat (Nickname: {nickname})</h2>
      <div
        style={{
          border: "1px solid black",
          height: 300,
          overflowY: "auto",
          padding: 10,
          marginBottom: 10,
        }}
      >
        {chatLog.map((msg, i) => (
          <div key={i}>
            <strong>{msg.sender}:</strong> {msg.message}
          </div>
        ))}
      </div>
      <input
        type="text"
        placeholder="Enter the message"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") sendMessage();
        }}
        style={{ width: "80%", marginRight: 10 }}
      />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}

export default App;

