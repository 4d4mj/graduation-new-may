"use client";

import React, { useCallback } from "react";
import SpeechRecognition, {
	useSpeechRecognition,
} from "react-speech-recognition";
import { Button } from "@/components/ui/button";
import { Icon } from "@/components/ui/icon";
import { Textarea } from "@/components/ui/textarea";

export default function ChatInput({ input, setInput, handleSubmit }) {
	/* ---------- speech‑recognition ---------- */
	const { transcript, listening, browserSupportsSpeechRecognition } =
		useSpeechRecognition();

	/* keep textarea in sync with live transcript */
	React.useEffect(() => {
		if (listening) setInput(transcript);
	}, [transcript, listening, setInput]);

	const toggleListening = useCallback(() => {
		if (!browserSupportsSpeechRecognition) return;
		if (listening) SpeechRecognition.stopListening();
		else SpeechRecognition.startListening({ interimResults: true });
	}, [listening, browserSupportsSpeechRecognition]);

	/* ---------- keyboard / form handlers ---------- */
	const onKeyDown = (e) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			handleSubmit(e);
		}
	};

	const onSubmit = (e) => {
		e.preventDefault();
		SpeechRecognition.stopListening();
		handleSubmit(e);
	};

	/* ---------- UI ---------- */
	return (
		<div className="max-w-3xl w-full mx-auto p-2">
			<form
				id="chat-form"
				onSubmit={onSubmit}
				className="bg-white rounded-lg overflow-hidden"
			>
				<Textarea
					value={input}
					onChange={(e) => setInput(e.target.value)}
					onKeyDown={onKeyDown}
					placeholder="Ask Anything"
					rows={1}
					className="w-full px-3 py-2 rounded-l-lg border-0 focus:border-0 focus-visible:border-0 focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0 shadow-none outline-none resize-none min-h-[2.5rem] max-h-[9rem] overflow-y-auto"
				/>
				<div className="flex justify-end p-2 gap-2">
					<Button
						type="button"
						variant="outline"
						onClick={toggleListening}
						disabled={!browserSupportsSpeechRecognition}
						className={!listening ? "bg-red-100 text-red-800" : ""}
					>
						<Icon>{listening ? "mic" : "mic_off"}</Icon>
					</Button>
					<Button type="submit">
						<Icon>send</Icon>
					</Button>
				</div>
			</form>

			{!browserSupportsSpeechRecognition && (
				<p className="text-xs text-center text-red-600 py-1">
					Voice input is not supported by your browser.
				</p>
			)}

			<p className="text-xs font-medium text-center py-1 text-gray-700">
				AI can make mistakes. Check important info.
			</p>
		</div>
	);
}
