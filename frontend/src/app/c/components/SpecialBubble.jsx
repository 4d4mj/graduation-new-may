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
import { useState } from "react";

export default function SpecialBubble({ message, setInput, addMessage }) {
	const [isDisabled, setIsDisabled] = useState(false);

	let payload;
	try {
		payload = JSON.parse(message.content);
	} catch {
		payload = null;
	}

	if (isDisabled) {
		return (
			<Card className="max-w-xl">
				<CardHeader>
					<CardTitle>Sent ✅</CardTitle>
					<CardDescription>
						Your response is being processed. Please wait.
					</CardDescription>
				</CardHeader>
			</Card>
		);
	}

	// Handle status 500 errors
	if (payload?.type === "error" && payload?.status === 500) {
		return (
			<Card className="max-w-xl border-red-200 bg-red-50">
				<CardHeader>
					<Badge
						variant="destructive"
						className="bg-red-600 text-white"
					>
						Server Error
					</Badge>
					<CardTitle className="text-red-800">
						Oops! Something went wrong
					</CardTitle>
					<CardDescription className="text-red-700">
						{payload.message ||
							"The server encountered an unexpected error. Please try again in a moment."}
					</CardDescription>
				</CardHeader>
				<CardFooter className="flex gap-2 justify-end">
					<Button
						variant="outline"
						className="border-red-300 text-red-700 hover:bg-red-100"
						onClick={() => {
							// Retry the last user message
							if (payload.retryMessage) {
								flushSync(() => setInput(payload.retryMessage));
								document
									.getElementById("chat-form")
									?.requestSubmit();
							}
						}}
					>
						Try Again
					</Button>
				</CardFooter>
			</Card>
		);
	}

	// Handle other error types (401, 403, 422, etc.)
	if (payload?.type === "error" && payload?.status) {
		const getErrorConfig = (status) => {
			switch (status) {
				case 401:
					return {
						title: "Authentication Required",
						description:
							"Your session has expired. Please log in again.",
						badge: "Unauthorized",
						color: "yellow",
						showRetry: false,
						showLogin: true,
					};
				case 403:
					return {
						title: "Access Denied",
						description:
							"You don't have permission to perform this action.",
						badge: "Forbidden",
						color: "orange",
						showRetry: false,
						showLogin: false,
					};
				case 422:
					return {
						title: "Invalid Request",
						description: "Please check your input and try again.",
						badge: "Validation Error",
						color: "blue",
						showRetry: true,
						showLogin: false,
					};
				default:
					return {
						title: "Error",
						description: "An unexpected error occurred.",
						badge: "Error",
						color: "red",
						showRetry: true,
						showLogin: false,
					};
			}
		};

		const config = getErrorConfig(payload.status);
		const colorClasses = {
			red: {
				card: "border-red-200 bg-red-50",
				badge: "bg-red-600",
				title: "text-red-800",
				desc: "text-red-700",
				button: "border-red-300 text-red-700 hover:bg-red-100",
			},
			yellow: {
				card: "border-yellow-200 bg-yellow-50",
				badge: "bg-yellow-600",
				title: "text-yellow-800",
				desc: "text-yellow-700",
				button: "border-yellow-300 text-yellow-700 hover:bg-yellow-100",
			},
			orange: {
				card: "border-orange-200 bg-orange-50",
				badge: "bg-orange-600",
				title: "text-orange-800",
				desc: "text-orange-700",
				button: "border-orange-300 text-orange-700 hover:bg-orange-100",
			},
			blue: {
				card: "border-blue-200 bg-blue-50",
				badge: "bg-blue-600",
				title: "text-blue-800",
				desc: "text-blue-700",
				button: "border-blue-300 text-blue-700 hover:bg-blue-100",
			},
		};
		const colors = colorClasses[config.color];

		return (
			<Card className={`max-w-xl ${colors.card}`}>
				<CardHeader>
					<Badge
						variant="destructive"
						className={`${colors.badge} text-white`}
					>
						{config.badge}
					</Badge>
					<CardTitle className={colors.title}>
						{config.title}
					</CardTitle>
					<CardDescription className={colors.desc}>
						{payload.message || config.description}
					</CardDescription>
				</CardHeader>
				{(config.showRetry || config.showLogin) && (
					<CardFooter className="flex gap-2 justify-end">
						{config.showLogin && (
							<Button
								variant="outline"
								className={colors.button}
								onClick={() => {
									// Redirect to login
									window.location.href = "/login";
								}}
							>
								Log In
							</Button>
						)}
						{config.showRetry && (
							<Button
								variant="outline"
								className={colors.button}
								onClick={() => {
									// Retry the last user message
									if (payload.retryMessage) {
										flushSync(() =>
											setInput(payload.retryMessage)
										);
										document
											.getElementById("chat-form")
											?.requestSubmit();
									}
								}}
							>
								Try Again
							</Button>
						)}
					</CardFooter>
				)}
			</Card>
		);
	}

	if (payload?.type === "booking_proposal") {
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
						Would you like to book an appointment with{" "}
						{payload.doctor_name} {payload.proposed_starts_at_display}?
					</CardDescription>
				</CardHeader>
				<CardFooter className="flex gap-2 justify-end">
					<Button
						variant="outline"
						onClick={() => {
							setIsDisabled(true);
							// Use flushSync to set input and submit form, consistent with other bubbles
							flushSync(() =>
								setInput(
									"No, I don't want to book this appointment."
								)
							);
							document
								.getElementById("chat-form")
								?.requestSubmit();
						}}
					>
						Cancel
					</Button>
					<Button
						onClick={() => {
							setIsDisabled(true);
							// Use flushSync to set input and submit form, consistent with other bubbles
							flushSync(() =>
								setInput("Yes, please book this appointment.")
							);
							document
								.getElementById("chat-form")
								?.requestSubmit();
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
						{payload.doctor} - {payload.specialty}
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
									setIsDisabled(true);
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
								setIsDisabled(true);
								// Use doctor ID instead of name in subsequent operations
								flushSync(() =>
									setInput(
										`I'd like to book with doctor_id ${doctor.id} (${doctor.name})`
									)
								);
								document
									.getElementById("chat-form")
									?.requestSubmit();
							}}
						>
							{doctor.name} · {doctor.specialty}
						</Button>
					))}
				</CardContent>
			</Card>
		);
	}
	// Handle new structured appointment confirmation
	if (payload?.type === "appointment_confirmed" && payload?.appointment_id) {
		return (
			<Card className="max-w-xl border-green-200 bg-green-50">
				<CardHeader>
					<Badge
						variant="secondary"
						className="bg-green-600 text-white"
					>
						{payload.agent || "Scheduler"}
					</Badge>
					<CardTitle className="text-green-800">
						🎉 Appointment Confirmed!
					</CardTitle>
					<CardDescription className="text-green-700">
						Your appointment with {payload.doctor_name} is confirmed
						for {payload.start_dt} at {payload.location}.
						{payload.notes && ` Reason: ${payload.notes}`}
					</CardDescription>
				</CardHeader>
				<CardContent className="space-y-3">
					<div className="text-sm text-green-600">
						<strong>Appointment ID:</strong> #
						{payload.appointment_id}
					</div>
					{payload.google_calendar_link && (
						<div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
							<p className="text-sm text-blue-700 mb-2">
								📅 Google Calendar Event
							</p>
							<Button
								variant="outline"
								size="sm"
								className="border-blue-300 text-blue-700 hover:bg-blue-100"
								onClick={() =>
									window.open(
										payload.google_calendar_link,
										"_blank"
									)
								}
							>
								Open in Google Calendar
							</Button>
						</div>
					)}
					{payload.google_calendar_invite_status &&
						!payload.google_calendar_link && (
							<div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
								Google Calendar:{" "}
								{payload.google_calendar_invite_status}
							</div>
						)}
				</CardContent>
			</Card>
		);
	}

	// Handle booking conflicts
	if (payload?.type === "booking_conflict") {
		return (
			<Card className="max-w-xl border-amber-200 bg-amber-50">
				<CardHeader>
					<Badge
						variant="secondary"
						className="bg-amber-600 text-white"
					>
						{payload.agent || "Scheduler"}
					</Badge>
					<CardTitle className="text-amber-800">
						⚠️ Time Slot Unavailable
					</CardTitle>
					<CardDescription className="text-amber-700">
						{payload.message || "This time slot is already booked."}
					</CardDescription>
				</CardHeader>
				<CardFooter>
					<Button
						variant="outline"
						className="border-amber-300 text-amber-700 hover:bg-amber-100"
						onClick={() => {
							flushSync(() =>
								setInput("Show me available times")
							);
							document
								.getElementById("chat-form")
								?.requestSubmit();
						}}
					>
						Show Available Times
					</Button>
				</CardFooter>
			</Card>
		);
	}

	// Handle booking errors
	if (payload?.type === "booking_error") {
		return (
			<Card className="max-w-xl border-red-200 bg-red-50">
				<CardHeader>
					<Badge
						variant="destructive"
						className="bg-red-600 text-white"
					>
						{payload.agent || "Scheduler"}
					</Badge>
					<CardTitle className="text-red-800">
						❌ Booking Failed
					</CardTitle>
					<CardDescription className="text-red-700">
						{payload.message ||
							"An error occurred while booking the appointment."}
					</CardDescription>
				</CardHeader>
				<CardFooter className="flex gap-2 justify-end">
					<Button
						variant="outline"
						className="border-red-300 text-red-700 hover:bg-red-100"
						onClick={() => {
							flushSync(() =>
								setInput("Help me book an appointment")
							);
							document
								.getElementById("chat-form")
								?.requestSubmit();
						}}
					>
						Try Again
					</Button>
				</CardFooter>
			</Card>
		);
	}

	// Legacy appointment confirmation (keeping for backward compatibility)
	if (payload?.status === "confirmed" && payload.id) {
		return (
			<Card className="max-w-xl">
				<CardHeader>
					<Badge
						variant="secondary"
						className="capitalize bg-green-600 text-primary-foreground"
					>
						Confirmation
					</Badge>
					<CardTitle>Appointment Confirmed</CardTitle>
					<CardDescription>
						Your appointment with {payload.doctor_name} is confirmed
						for {payload.start_dt}.
					</CardDescription>
				</CardHeader>
			</Card>
		);
	}

	// New handler for "no_doctors" type
	if (payload?.type === "no_doctors") {
		return (
			<Card className="max-w-xl">
				<CardHeader>
					<Badge
						variant="secondary"
						className="bg-gray-600 text-primary-foreground"
					>
						Scheduler
					</Badge>
					<CardTitle>No Doctors Found</CardTitle>
					<CardDescription>{payload.message}</CardDescription>
				</CardHeader>
			</Card>
		);
	}

	return <ChatBubble message={message} />;
}
