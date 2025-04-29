"use server";

import settings from "@/config/settings";
import { redirect } from "next/navigation";
import { registerSchema } from "@/schemas/register"; // your Zod form schema

export async function register(formData) {
	// 1. validate UI input
	const result = registerSchema.safeParse(formData);
	if (!result.success) {
		return { errors: result.error.flatten().fieldErrors };
	}

	// 2. build payload for the API
	const {
		repeat_password, // discard
		sex,
		first_name,
		last_name,
		dob,
		phone,
		address,
		...rest // email, password, optional role
	} = result.data;

	const payload = {
		...rest,
		patient_profile: {
			sex,
			first_name,
			last_name,
			dob: dob.toISOString().slice(0, 10), // YYYY-MM-DD
			phone,
			address,
		},
	};

	// 3. send to FastAPI
	const res = await fetch(`${settings.apiInternalUrl}/auth/register`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		credentials: "include",
		body: JSON.stringify(payload),
	});

	if (!res.ok) {
		const err = await res.json().catch(() => ({}));
		return { errors: { api: [err.detail || res.statusText] } };
	}

	redirect("/login");
}
