// src/schemas/register_schema.ts
import { z } from "zod";


export const registerSchema = z
	.object({
		email: z.string().trim().email({ message: "Invalid email address" }),
		password: z.string().trim().min(8, {
			message: "Password must be at least 8 characters",
		}),
		first_name: z
			.string()
			.trim()
			.min(1, { message: "First name is required" }),
		last_name: z
			.string()
			.trim()
			.min(1, { message: "Last name is required" }),
		sex: z.enum(["M", "F"]),
		phone: z.string().trim().min(1, { message: "Phone is required" }),
		address: z.string().trim().min(1, { message: "Address is required" }),
		dob: z.coerce.date({
			required_error: "Date of birth is required",
			invalid_type_error: "Invalid date",
		}),
		role: z.enum(["doctor", "patient", "admin"]).optional(),
		repeat_password: z
			.string()
			.trim()
			.min(8, {
				message: "Repeat password must be at least 8 characters",
			}),
	})
	.strict()
	.refine((data) => data.password === data.repeat_password, {
		path: ["repeat_password"],
		message: "Passwords do not match",
	});
