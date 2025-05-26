import { getUser } from "@/lib/user";
import { redirect } from "next/navigation";
import Chat from "./components/Chat";
import { SidebarProvider } from "@/components/ui/sidebar";
import ChatSideBar from "./components/Sidebar";

export default async function Page() {
	const user = await getUser();
	console.log("user", user);

	if (!user) {
		console.log("No user found, redirecting to login...");
		// redirect("/login");
	}

	return (
		<SidebarProvider defaultOpen={false}>
			<ChatSideBar />
			<Chat user={user} />
		</SidebarProvider>
	);
}
