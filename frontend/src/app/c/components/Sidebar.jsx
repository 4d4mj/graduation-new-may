import {
	Sidebar,
	SidebarContent,
	SidebarGroup,
	SidebarHeader,
} from "@/components/ui/sidebar";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import settings from "@/config/settings";
import { cookies } from "next/headers";
import { revalidatePath } from "next/cache"; // Added for server action
import { formatInTimeZone } from "date-fns-tz"; // Import formatInTimeZone
import { Icon } from "@/components/ui/icon";

async function fetchAppointments() {
	const cookieStore = await cookies();
	const token = cookieStore.get("session")?.value;

	if (!token) {
		return []; // Return an empty array if no token is found
	}

	const response = await fetch(`${settings.apiInternalUrl}/appointments`, {
		headers: {
			Authorization: `Bearer ${token}`,
		},
	});

	if (!response.ok) {
		throw new Error("Failed to fetch appointments");
	}

	return response.json();
}

// Server Action to cancel an appointment
async function cancelAppointment(formData) {
	"use server";
	const appointmentId = formData.get("appointmentId");

	if (!appointmentId) {
		console.error("Appointment ID is missing for cancellation.");
		// Consider returning a response or throwing an error for client-side handling
		return;
	}

	const cookieStore = await cookies();
	const token = cookieStore.get("session")?.value;

	if (!token) {
		console.error("Authorization token is missing for cancel action.");
		// Consider returning a response or throwing an error
		return;
	}

	try {
		const response = await fetch(
			// Ensure settings.apiInternalUrl is accessible here or pass it if needed
			// For this example, assuming 'settings' is available in this scope as in fetchAppointments
			`${settings.apiInternalUrl}/appointments/${appointmentId}`,
			{
				method: "DELETE",
				headers: {
					Authorization: `Bearer ${token}`,
				},
			}
		);

		if (!response.ok) {
			const errorText = await response.text();
			console.error(
				`Failed to cancel appointment ${appointmentId}: ${response.status} ${errorText}`
			);
			// Consider returning a response or throwing an error
			return;
		}

		// Revalidate the path to refresh the appointments list.
		// Adjust '/c' if your appointments are displayed on a different base path.
		// Using 'layout' can help if the data is used in a layout.
		revalidatePath("/c", "layout");
	} catch (error) {
		console.error(`Error cancelling appointment ${appointmentId}:`, error);
		// Consider returning a response or throwing an error
	}
}

function DoctorSideBarItem({ appointment }) {
	// New component for doctor's view
	const timeZone = "Asia/Beirut";
	const startTime = formatInTimeZone(
		new Date(appointment.starts_at),
		timeZone,
		"HH:mm"
	);
	const endTime = formatInTimeZone(
		new Date(appointment.ends_at),
		timeZone,
		"HH:mm"
	);
	const appointmentDate = formatInTimeZone(
		new Date(appointment.starts_at),
		timeZone,
		"MMMM do, yyyy"
	);

	return (
		<Alert>
			<AlertTitle>
				Patient: {appointment.patient_name}
			</AlertTitle>
			<AlertDescription>
				<div className="flex gap-2 items-center">
					<Icon>schedule</Icon>
					<div>
						<p>{appointmentDate}</p>
						<p>
							{startTime} - {endTime}
						</p>
					</div>
				</div>
				<div className="flex gap-2 items-center">
					<Icon>location_on</Icon>
					<p>{appointment.location}</p>
				</div>
				<div className="flex gap-2 items-center">
					<Icon>description</Icon>
					<p className="text-sm text-gray-500">{appointment.notes}</p>
				</div>
				{/* No cancel button for doctors in this version */}
			</AlertDescription>
		</Alert>
	);
}

function PatientSideBarItem({ appointment }) {
	// Renamed from SideBarItem
	const timeZone = "Asia/Beirut"; // Example timezone, change as needed
	const startTime = formatInTimeZone(
		new Date(appointment.starts_at),
		timeZone,
		"HH:mm"
	);
	const endTime = formatInTimeZone(
		new Date(appointment.ends_at),
		timeZone,
		"HH:mm"
	);
	const appointmentDate = formatInTimeZone(
		new Date(appointment.starts_at),
		timeZone,
		"MMMM do, yyyy"
	);

	return (
		<Alert>
			<AlertTitle>
				Dr. {appointment.doctor_profile?.first_name}{" "}
				{appointment.doctor_profile?.last_name}
			</AlertTitle>
			<AlertDescription>
				<div className="flex gap-2 items-center">
					<Icon>schedule</Icon>
					<div>
						<p>{appointmentDate}</p>
						<p>
							{startTime} - {endTime}
						</p>
					</div>
				</div>
				<div className="flex gap-2 items-center">
					<Icon>location_on</Icon>
					<p>{appointment.location}</p>
				</div>
				<div className="flex gap-2 items-center">
					<Icon>description</Icon>
					<p className="text-sm text-gray-500">{appointment.notes}</p>
				</div>
				<form action={cancelAppointment} className="mt-2">
					<input
						type="hidden"
						name="appointmentId"
						value={appointment.id}
					/>
					<Button type="submit" variant={"destructive"} size="sm">
						Cancel
					</Button>
				</form>
			</AlertDescription>
		</Alert>
	);
}

export default async function ChatSideBar({ user }) {
	const appointments = await fetchAppointments();
	const userRole = user?.role;

	console.log("User Role:", userRole);

	// console.log(appointments);

	return (
		<Sidebar>
			<SidebarHeader>
				<h3 className="font-semibold text-white">
					Upcoming appointments
				</h3>
			</SidebarHeader>
			<SidebarContent className={"px-2"}>
				<SidebarGroup className={"space-y-2"}>
					{appointments.map((appointment) =>
						userRole === "doctor" ? (
							<DoctorSideBarItem
								key={appointment.id}
								appointment={appointment}
							/>
						) : (
							<PatientSideBarItem
								key={appointment.id}
								appointment={appointment}
							/>
						)
					)}
				</SidebarGroup>
			</SidebarContent>
		</Sidebar>
	);
}
