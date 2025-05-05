import { Badge } from "@/components/ui/badge";
import ReactMarkdown from "react-markdown";

export default function ChatBubble({ message }) {
	const { role, content, agent } = message;
	const isUser = role === "user";

	return (
		<div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
			<div
				className={`max-w-[70%] py-2 rounded-3xl space-y-1 ${
					isUser ? "bg-gray-600 text-primary-foreground px-4" : ""
				}`}
			>
				<ReactMarkdown>{content}</ReactMarkdown>
				{agent && (
					<Badge
						variant="secondary"
						className="capitalize bg-gray-600 text-primary-foreground"
					>
						{agent}
					</Badge>
				)}
			</div>
		</div>
	);
}
