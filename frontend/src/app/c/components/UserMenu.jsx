import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { logout } from "../actions";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuGroup,
	DropdownMenuItem,
	DropdownMenuLabel,
	DropdownMenuSeparator,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";

export function UserMenu({ user }) {
	function getInitials(user) {
		const firstNameInitial = user.patient_profile?.first_name?.[0] || "";
		const lastNameInitial = user.patient_profile?.last_name?.[0] || "";
		return `${firstNameInitial}${lastNameInitial}`.toUpperCase();
	}

	return (
		<DropdownMenu>
			<DropdownMenuTrigger asChild>
				<Button variant="ghost">
					{user.patient_profile?.first_name}
					<Avatar>
						<AvatarFallback>{getInitials(user)}</AvatarFallback>
					</Avatar>
				</Button>
			</DropdownMenuTrigger>
			<DropdownMenuContent className="w-56">
				<DropdownMenuLabel className="flex justify-between">
					My Account
					<Badge
						variant="outline"
						className="capitalize bg-green-200"
					>
						{user.role}
					</Badge>
				</DropdownMenuLabel>

				<DropdownMenuSeparator />
				<DropdownMenuGroup>
					<DropdownMenuItem>Profile</DropdownMenuItem>
					<DropdownMenuItem>Settings</DropdownMenuItem>
				</DropdownMenuGroup>
				<DropdownMenuSeparator />
				<form action={logout} className="w-full">
					<button
						type="submit"
						variant="destructive"
						className="w-full"
					>
						<DropdownMenuItem variant="destructive">
							Log out
						</DropdownMenuItem>
					</button>
				</form>
			</DropdownMenuContent>
		</DropdownMenu>
	);
}
