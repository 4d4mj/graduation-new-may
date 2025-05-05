import { useEffect, useRef } from "react";
import ChatBubble from "./ChatBubble";

export default function ConversationArea({ user, messages }) {
	const endRef = useRef(null);

	// Scroll to bottom on new message
	useEffect(() => {
		endRef.current?.scrollIntoView({ behavior: "smooth" });
	}, [messages]);

	if (!messages || messages.length === 0) {
		return (
			<div className="text-center flex-1 bg-gradient-to-r from-slate-800 via-slate-600 to-slate-700 text-transparent bg-clip-text content-center">
				<h1 className="font-bold text-4xl mb-2">
					Greetings, {user.patient_profile.first_name}
				</h1>
				<p className="text-2xl">
					How can I assist you with your health needs today?
				</p>
			</div>
		);
	}

	return (
		<div className="flex-1 w-full overflow-y-auto py-2 space-y-2">
			<div className="w-full px-4 md:px-0 md:w-lg lg:w-xl xl:w-3xl mx-auto">
				{messages.map((message, index) => (
					<ChatBubble key={index} message={message} />
				))}
				<div ref={endRef} />
			</div>
		</div>
	);
}
