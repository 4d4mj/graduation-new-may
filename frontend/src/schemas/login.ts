import { z } from "zod";

export const loginSchema = z
	.object({
		email: z.string().trim().email({ message: "Invalid email address" }),
		password: z
			.string()
			.trim()
			.min(8, { message: "Password must be at least 8 characters" })
			.max(128, {
				message: "Password must be at most 128 characters",
			}),
	})
	.strict();
