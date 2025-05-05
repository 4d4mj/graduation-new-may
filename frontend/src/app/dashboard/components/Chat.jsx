"use client";

import { useState } from "react";
import ConversationArea from "./ConversationArea";
import ChatInput from "./ChatInput";
import Navbar from "./Navbar";

export default function Chat({ user }) {
	const [input, setInput] = useState("");
	const [messages, setMessages] = useState([]); // [{role, content}, …]

	const handleSubmit = async (e) => {
		e.preventDefault();
		if (!input.trim()) return;

		// 1) add the user message locally
		const userMsg = { role: "user", content: input };
		const newHistory = [...messages, userMsg];
		setMessages(newHistory);

		const historyPayload = messages; // our prior messages
		const payload = {
			message: input,
			history: historyPayload,
		};

		setInput("");

		try {
			// 2) call your backend; credentials: 'include' to send the session cookie
			console.log("sending payload", payload);
			const res = await fetch(
				`${
					process.env.NEXT_PUBLIC_API_URL || "http://backend:8000"
				}/chat/test`,
				{
					method: "POST",
					headers: { "Content-Type": "application/json" },
					credentials: "include",
					body: JSON.stringify(payload),
				}
			);
			const data = await res.json();
			console.log(data);

			// 3) append the assistant's reply
			const assistantMsg = {
				role: "assistant",
				content: data.reply,
				agent: data.agent,
			};
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
		<main className="grow flex flex-col items-center h-screen max-h-screen w-full bg-gray-400">
			<Navbar {...{ user }} />
			<ConversationArea {...{user, messages}} />
			<ChatInput {...{ input, setInput, handleSubmit }} />
		</main>
	);
}
