import { getUser } from "@/lib/user";
import Chat from "./components/Chat";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { ChatSideBar } from "./components/SideBar";

export default async function Page() {
	const user = await getUser();
	console.log("user", user);
	// if (!user) redirect("/login");
	if (!user) {
		console.log("No user found, redirecting to login...");
		return null;
	}

	return (
		<SidebarProvider defaultOpen={false}>
			<ChatSideBar />
				<Chat user={user} />
		</SidebarProvider>
	);
}
