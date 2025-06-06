import { useEffect, useRef } from "react";
import SpecialBubble from "./SpecialBubble";
import ThinkingBubble from "./ThinkingBubble";

export default function ConversationArea({
	user,
	setInput,
	messages,
	addMessage,
	isThinking,
}) {
	const endRef = useRef(null);
	// Scroll to bottom on new message or thinking state change
	useEffect(() => {
		endRef.current?.scrollIntoView({ behavior: "smooth" });
	}, [messages, isThinking]);

	if (!messages || messages.length === 0) {
		return (
			<div className="text-center flex-1 bg-gradient-to-r from-slate-800 via-slate-600 to-slate-700 text-transparent bg-clip-text content-center">
				<h1 className="font-bold text-4xl mb-2">
					Greetings,{" "}
					{user?.patient_profile
						? user.patient_profile.first_name
						: user?.doctor_profile
						? `Dr. ${user.doctor_profile.first_name}`
						: "Guest"}
				</h1>
				<p className="text-2xl">
					How can I assist you with your health needs today?
				</p>
			</div>
		);
	}

	return (
		<div className="flex-1 w-full overflow-y-auto py-2 space-y-2">
			<div className="w-full px-4 md:px-0 md:w-lg lg:w-xl xl:w-3xl mx-auto space-y-4">
				{messages.map((message, index) => (
					<SpecialBubble
						key={index}
						message={message}
						setInput={setInput}
						addMessage={addMessage}
					/>
				))}
				{isThinking && <ThinkingBubble />}
				<div ref={endRef} />
			</div>
		</div>
	);
}
