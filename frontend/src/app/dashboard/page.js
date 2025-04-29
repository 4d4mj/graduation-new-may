import { redirect } from "next/navigation";
import { getUser } from "@/lib/user";
import Dashboard from "./Dashboard";

export default async function Page() {
	const user = await getUser();
	console.log("user", user);
	// if (!user) redirect("/login");
	if (!user) {
		console.log("No user found, redirecting to login...");
		return null;
	}

	return (
		<Dashboard user={user} />
	);
}
