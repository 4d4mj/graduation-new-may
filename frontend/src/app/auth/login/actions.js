// "use server";

import { redirect } from "next/navigation";
import { loginSchema } from "@/schemas/login";
import settings from "@/config/settings";

export async function login(formData) {
	const result = loginSchema.safeParse(Object.fromEntries(formData));

	if (!result.success) {
		return {
			errors: result.error.flatten().fieldErrors,
		};
	}

	const { email, password } = result.data;

	const res = await fetch(`${settings.apiInternalUrl}/auth/login`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ email, password }),
		credentials: "include",
	});

	if (!res.ok) {
		const body = await res.json().catch(() => ({}));
		return { errors: body.detail || "Invalid credentials" };
	}

	redirect("/c");
}
