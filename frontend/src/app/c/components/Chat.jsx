"use client";

import { useState } from "react";
import ConversationArea from "./ConversationArea";
import dynamic from "next/dynamic";
import Navbar from "./Navbar";
import { sendChat } from "../actions";

const ChatInput = dynamic(() => import("./ChatInput"), { ssr: false });

export default function Chat({ user }) {
	const [input, setInput] = useState("");
	const [messages, setMessages] = useState([]);
	const [isThinking, setIsThinking] = useState(false);
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

		// Start thinking state and timer
		setIsThinking(true);
		try {
			console.log("sending payload", payload);

			// Use the sendChat action instead of direct fetch
			const result = await sendChat(payload);

			console.log(result);

			// Check if the request was successful
			if (!result.success) {
				// Handle error response
				const { error } = result;
				let errorMessage = error.message;

				// Handle specific error types
				if (error.type === "AUTHENTICATION_FAILED") {
					// Redirect to login after a short delay
					setTimeout(() => {
						window.location.href = "/login";
					}, 3000);
				}

				const errorPayload = {
					type: "error",
					status: error.status,
					message: errorMessage,
					retryMessage: payload.message
				};

				setMessages((m) => [
					...m,
					{ role: "assistant", content: JSON.stringify(errorPayload) },
				]);
				return;
			}

			// Success - append the assistant's reply
			const data = result.data;
			const assistantMsg = {
				role: "assistant",
				content: data.reply,
				agent: data.agent,
				interrupt_id: data.interrupt_id, // Store the interrupt ID if present
				thinkingTime: data.thinking_time, // Use thinking time from backend
			};
			setMessages((m) => [...m, assistantMsg]);
		} catch (err) {
			console.warn("Chat error:", err);

			// Handle unexpected errors (shouldn't happen with new structure, but just in case)
			const errorPayload = {
				type: "error",
				status: 500,
				message: "An unexpected error occurred. Please try again.",
				retryMessage: payload.message
			};

			setMessages((m) => [
				...m,
				{ role: "assistant", content: JSON.stringify(errorPayload) },
			]);
		}finally {
			// Stop thinking state
			setIsThinking(false);
		}
	};

	return (
		<main className="grow flex flex-col items-center h-screen max-h-screen w-full">
			<Navbar {...{ user }} />			<ConversationArea
				{...{ user, setInput, messages, isThinking }}
				addMessage={(m) => setMessages((old) => [...old, m])}
			/>
			<ChatInput {...{ input, setInput, handleSubmit }} />
		</main>
	);
}
