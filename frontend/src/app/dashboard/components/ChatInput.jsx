import { Button } from "@/components/ui/button";
import { Icon } from "@/components/ui/icon";
import { Textarea } from "@/components/ui/textarea";
import React from "react"; // Make sure React is imported if not already globally available

export default function ChatInput({ input, setInput, handleSubmit }) {
	// Optional: Handle Enter key submission (Shift+Enter for newline)
	const handleKeyDown = (e) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault(); // Prevent default newline insertion
			handleSubmit(e); // Trigger form submission
		}
	};

	return (
		<div className="w-full px-4 md:px-0 md:w-lg lg:w-xl xl:w-3xl">
			<form
				onSubmit={handleSubmit}
				className="bg-form rounded-3xl overflow-hidden p-2"
			>
				<Textarea
					value={input}
					onChange={(e) => setInput(e.target.value)}
					onKeyDown={handleKeyDown}
					placeholder="Ask Anything"
					rows="1" // Start with 1 row visible
					className="min-h-0 text-base md:text-base w-full border-0 focus:border-0 focus-visible:border-0 focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0 shadow-none outline-none resize-none [field-sizing:content] max-h-[9rem] overflow-y-auto" // Override all borders and outlines
				/>
				<div className="flex justify-end p-2 gap-2">
					<Button type="submit" variant="outline"><Icon>mic</Icon></Button>
					<Button type="submit"><Icon>send</Icon></Button>
				</div>
			</form>
			<p className="text-xs font-medium text-center py-1 text-gray-700">
				AI can make mistakes. Check important info.
			</p>
		</div>
	);
}
