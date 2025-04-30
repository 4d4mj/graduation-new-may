"use client";

import { useState, useRef, useEffect } from "react";

export default function ChatBox() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]); // [{role, content}, …]
  const endRef = useRef(null);

  // scroll to bottom on new message
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // 1) add the user message locally
    const userMsg = { role: "user", content: input };
    setMessages((m) => [...m, userMsg]);

    const historyPayload = messages; // our prior messages
    const payload = {
      message: input,
      history: historyPayload,
    };

    setInput("");

    try {
      // 2) call your backend; credentials: 'include' to send the session cookie
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://backend:8000"}/chat/`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify(payload),
        }
      );
      const data = await res.json();

      // 3) append the assistant’s reply
      const assistantMsg = { role: "assistant", content: data.response };
      setMessages((m) => [...m, assistantMsg]);
    } catch (err) {
      console.error("Chat error:", err);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: "⚠️ Something went wrong." },
      ]);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-2 space-y-2 border rounded bg-white">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[70%] px-3 py-1 rounded ${
                m.role === "user" ? "bg-blue-200" : "bg-gray-200"
              }`}
            >
              {m.content}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form onSubmit={handleSubmit} className="mt-2 flex">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message…"
          className="flex-1 border rounded-l px-3 py-2 focus:outline-none"
        />
        <button
          type="submit"
          className="px-4 py-2 bg-blue-600 text-white rounded-r hover:bg-blue-700"
        >
          Send
        </button>
      </form>
    </div>
  );
}
