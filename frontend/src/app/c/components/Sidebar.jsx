import {
	Sidebar,
	SidebarContent,
	SidebarGroup,
	SidebarHeader,
} from "@/components/ui/sidebar";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";

const appointmentData = [
	{
		id: 1,
		doctor_profile: { firstname: "John" },
		start_time: "10:00 AM",
		end_time: "11:00 AM",
		location: "Room 101",
		notes: "Follow-up on blood test results",
	},
	{
		id: 2,
		doctor_profile: { firstname: "Jane" },
		start_time: "11:30 AM",
		end_time: "12:30 PM",
		location: "Room 102",
		notes: "Discuss medication side effects",
	},
	{
		id: 3,
		doctor_profile: { firstname: "Alice" },
		start_time: "1:00 PM",
		end_time: "2:00 PM",
		location: "Room 103",
		notes: "Annual check-up",
	},
	{
		id: 4,
		doctor_profile: { firstname: "Bob" },
		start_time: "2:30 PM",
		end_time: "3:30 PM",
		location: "Room 104",
		notes: "Review treatment plan",
	},
	{
		id: 5,
		doctor_profile: { firstname: "Charlie" },
		start_time: "4:00 PM",
		end_time: "5:00 PM",
		location: "Room 105",
		notes: "Discuss test results",
	},
	{
		id: 6,
		doctor_profile: { firstname: "David" },
		start_time: "5:30 PM",
		end_time: "6:30 PM",
		location: "Room 106",
		notes: "Follow-up on treatment",
	},
	{
		id: 7,
		doctor_profile: { firstname: "Eve" },
		start_time: "7:00 PM",
		end_time: "8:00 PM",
		location: "Room 107",
		notes: "Discuss lifestyle changes",
	},
	{
		id: 8,
		doctor_profile: { firstname: "Frank" },
		start_time: "8:30 PM",
		end_time: "9:30 PM",
		location: "Room 108",
		notes: "Review lab results",
	},
];

function SideBarItem({ appointment }) {
	return (
		<Alert>
			<AlertTitle>{appointment.doctor_profile?.firstname}</AlertTitle>
			<AlertDescription>
				<p>
					{appointment.start_time} - {appointment.end_time} -{" "}
					{appointment.location} - {appointment.notes}
				</p>
				<Button variant={"destructive"}>
					Cancel
				</Button>
			</AlertDescription>
		</Alert>
	);
}

export function ChatSideBar() {
	console.log("hello");
	return (
		<Sidebar>
			<SidebarHeader>
				<h3 className="font-semibold text-white">
					Upcoming appointments
				</h3>
			</SidebarHeader>
			<SidebarContent className={"px-2"}>
				<SidebarGroup className={"space-y-2 bg-slate-"}>
					{appointmentData.map((appointment) => (
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
