"use client";

import { useEffect, useState } from "react";

export default function ThinkingBubble() {
	const [isVisible, setIsVisible] = useState(false);
	const text = "AI is thinking...";
	useEffect(() => {
		// Start the animation immediately
		setIsVisible(true);
	}, []);
	return (
		<div className="flex justify-start">
			<div className="relative">
				{/* Main text with shine effect and rise animation */}
				<div
					className={`font-medium text-sm bg-gradient-to-r from-black via-gray-500 to-black bg-clip-text text-transparent animate-shine transition-all duration-700 ${
						isVisible
							? "transform translate-y-0 opacity-100"
							: "transform translate-y-4 opacity-0"
					}`}
				>
					{text}
				</div>
			</div>
		</div>
	);
}
