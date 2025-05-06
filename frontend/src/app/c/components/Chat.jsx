"use client";

import { useState } from "react";
import ConversationArea from "./ConversationArea";
import dynamic from "next/dynamic";
import Navbar from "./Navbar";
import settings from "@/config/settings";

const ChatInput = dynamic(() => import("./ChatInput"), { ssr: false });

export default function Chat({ user }) {
	const [input, setInput] = useState("");
	const [messages, setMessages] = useState([]);

	const handleSubmit = async (e) => {
		e.preventDefault();
		if (!input.trim()) return;

		// 1) add the user message locally
		const userMsg = { role: "user", content: input };
		const newHistory = [...messages, userMsg];
		setMessages(newHistory);

		const payload = {
			message: input,
			user_tz: Intl.DateTimeFormat().resolvedOptions().timeZone,
		};

		setInput("");

		try {
			console.log("sending payload", payload);
			const res = await fetch(
				`${settings.apiUrl || "http://backend:8000"}/chat/`,
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
				interrupt_id: data.interrupt_id, // Store the interrupt ID if present
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
			<ConversationArea
				{...{ user, setInput, messages }}
				addMessage={(m) => setMessages((old) => [...old, m])}
			/>
			<ChatInput {...{ input, setInput, handleSubmit }} />
		</main>
	);
}
