import { SidebarTrigger } from "@/components/ui/sidebar";
import { UserMenu } from "./UserMenu";

export default function Navbar({ user }) {
	return (
		<nav className="flex justify-between items-center sticky top-0 p-3 z-50 w-full">
			<SidebarTrigger />
			<UserMenu {...{user}} />
		</nav>
	);
}
