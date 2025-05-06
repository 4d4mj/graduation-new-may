import { Button } from "@/components/ui/button";
import { Icon } from "@/components/ui/icon";
import Link from "next/link";

export default function Page() {
	return (
		<div className="h-screen w-screen flex flex-col items-center justify-center bg-gray-400">
			<Icon size="xl">warning</Icon>
			<p className="mb-2">Page not found</p>
			<Button>
				<Link href="/">Back Home</Link>
			</Button>
		</div>
	);
}
