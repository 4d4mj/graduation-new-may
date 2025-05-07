import ChatBubble from "./ChatBubble";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
	CardFooter,
} from "@/components/ui/card";
import { flushSync } from "react-dom";
import { Badge } from "@/components/ui/badge";
import { sendChat } from "../actions";

export default function SpecialBubble({ message, setInput, addMessage }) {
	let payload;
	try {
		payload = JSON.parse(message.content);
	} catch {
		payload = null;
	}

	if (
		payload?.type === "confirm_booking" &&
		payload.doctor &&
		payload.starts_at
	) {
		// Format the date and time for display
		const appointmentDate = new Date(payload.starts_at);
		const formattedDate = appointmentDate.toLocaleDateString(undefined, {
			weekday: "long",
			year: "numeric",
			month: "long",
			day: "numeric",
		});
		const formattedTime = appointmentDate.toLocaleTimeString(undefined, {
			hour: "2-digit",
			minute: "2-digit",
		});

		return (
			<Card className="max-w-xl">
				<CardHeader>
					<Badge
						variant="secondary"
						className="capitalize bg-gray-600 text-primary-foreground"
					>
						Scheduler
					</Badge>
					<CardTitle>Confirm Your Appointment</CardTitle>
					<CardDescription>
						Would you like to book an appointment with Dr.{" "}
						{payload.doctor} on {formattedDate} at {formattedTime}?
					</CardDescription>
				</CardHeader>
				<CardFooter className="flex gap-2 justify-end">
					<Button
						variant="outline"
						onClick={async () => {
							// Send "no" response back with the interrupt ID
							const res = await sendChat({
								message:
									"No, I don't want to book this appointment.",
								interrupt_id: message.interrupt_id,
								resume_value: "no",
							});

							addMessage?.({
								role: "user",
								content: "No, I don't want to book this appointment.",
							});

							// Add the assistant's response to the messages
							addMessage?.({
								role: "assistant",
								content: res.reply,
								agent: res.agent,
							});
						}}
					>
						Cancel
					</Button>
					<Button
						onClick={async () => {
							// Send "yes" response back with the interrupt ID
							const res = await sendChat({
								message: "Yes, please book this appointment.",
								interrupt_id: message.interrupt_id,
								resume_value: "yes",
							});

							addMessage?.({
								role: "user",
								content: "Yes, please book this appointment.",
							});

							// Add the assistant's response to the messages
							addMessage?.({
								role: "assistant",
								content: res.reply,
								agent: res.agent,
							});
						}}
					>
						Book It ✅
					</Button>
				</CardFooter>
			</Card>
		);
	}

	if (payload?.type === "slots" && Array.isArray(payload.options)) {
		return (
			<Card className="max-w-xl">
				<CardHeader>
					{payload.agent && (
						<Badge
							variant="secondary"
							className="capitalize bg-gray-600 text-primary-foreground"
						>
							{payload.agent}
						</Badge>
					)}
					<CardTitle>Please Select an Appointment Slot</CardTitle>
					<CardDescription>
						{payload.doctor} on {payload.date}
					</CardDescription>
				</CardHeader>
				<CardContent>
					{/* 3 equal columns on phones, 4 on ≥640 px */}
					<div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
						{payload.options.map((opt) => (
							<Button
								key={opt}
								type="button"
								className="w-full" /* make every btn fill its grid cell */
								onClick={() => {
									// synchronously update the input state before submitting
									flushSync(() =>
										setInput(payload.reply_template + opt)
									);
									document
										.getElementById("chat-form")
										?.requestSubmit();
								}} /* set input and immediately submit */
							>
								{opt}
							</Button>
						))}
					</div>
				</CardContent>
			</Card>
		);
	}

	// add just after the "slots" handler
	if (payload?.type === "doctors" && Array.isArray(payload.doctors)) {
		return (
			<Card className="max-w-xl">
				<CardHeader>
					<Badge
						variant="secondary"
						className="bg-gray-600 text-primary-foreground"
					>
						{payload.agent ?? "Scheduler"}
					</Badge>
					<CardTitle>Select a doctor</CardTitle>
					<CardDescription>{payload.message}</CardDescription>
				</CardHeader>
				<CardContent className="grid gap-2">
					{payload.doctors.map((doctor) => (
						<Button
							key={doctor.id}
							onClick={() => {
								// Use doctor ID instead of name in subsequent operations
								flushSync(() =>
									setInput(
										`I'd like to book with doctor_id ${doctor.id} (${doctor.name})`
									)
								);
								document.getElementById("chat-form")?.requestSubmit();
							}}
						>
							{doctor.name} · {doctor.specialty}
						</Button>
					))}
				</CardContent>
			</Card>
		);
	}

	return <ChatBubble message={message} />;
}
