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

async function fetchAppointments() {
	const cookieStore = await cookies();
	const token = cookieStore.get("session")?.value;

	if (!token) {
		throw new Error("Authorization token is missing");
	}

	const response = await fetch(
		`${settings.apiInternalUrl}/appointments`,
		{
			headers: {
				Authorization: `Bearer ${token}`,
			},
		}
	);

	if (!response.ok) {
		throw new Error("Failed to fetch appointments");
	}

	return response.json();
}

// Server Action to cancel an appointment
async function cancelAppointment(formData) {
  'use server';
  const appointmentId = formData.get('appointmentId');

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
      console.error(`Failed to cancel appointment ${appointmentId}: ${response.status} ${errorText}`);
      // Consider returning a response or throwing an error
      return;
    }

    // Revalidate the path to refresh the appointments list.
    // Adjust '/c' if your appointments are displayed on a different base path.
    // Using 'layout' can help if the data is used in a layout.
    revalidatePath('/c', 'layout');
  } catch (error) {
    console.error(`Error cancelling appointment ${appointmentId}:`, error);
    // Consider returning a response or throwing an error
  }
}

function SideBarItem({ appointment }) {
	return (
		<Alert>
			<AlertTitle>{appointment.patient_id} Dr. {appointment.doctor_profile?.first_name} {appointment.doctor_profile?.last_name}</AlertTitle>
			<AlertDescription>
				<p>
					{appointment.starts_at} - {appointment.ends_at} -{" "}
					{appointment.location} - {appointment.notes}
				</p>
				<form action={cancelAppointment}>
					<input type="hidden" name="appointmentId" value={appointment.id} />
					<Button type="submit" variant={"destructive"}>Cancel</Button>
				</form>
			</AlertDescription>
		</Alert>
	);
}

export default async function ChatSideBar() {
	const appointments = await fetchAppointments();

	console.log(appointments);

	return (
		<Sidebar>
			<SidebarHeader>
				<h3 className="font-semibold text-white">
					Upcoming appointments
				</h3>
			</SidebarHeader>
			<SidebarContent className={"px-2"}>
				<SidebarGroup className={"space-y-2 bg-slate-"}>
					{appointments.map((appointment) => (
						<SideBarItem
							key={appointment.id}
							appointment={appointment}
						/>
					))}
				</SidebarGroup>
			</SidebarContent>
		</Sidebar>
	);
}
