"use client";

import * as React from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useFormStatus } from "react-dom";
import { loginSchema } from "@/schemas/login";
import { login } from "./actions";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
	Form,
	FormControl,
	FormField,
	FormItem,
	FormLabel,
	FormMessage,
} from "@/components/ui/form";
import { toast } from "sonner";

export function LoginForm({ className, ...props }) {
	const { pending } = useFormStatus();

	// Initialize form with RHF and Zod
	const form = useForm({
		resolver: zodResolver(loginSchema),
		defaultValues: { email: "", password: "" },
	});

	// Handle form submission with RHF to show Zod errors
	const onSubmit = async (data) => {
		const formData = new FormData();
		Object.entries(data).forEach(([key, value]) => {
			formData.append(key, value);
		});
		const result = await login(formData);
		if (result.errors) {
			toast.error("Invalid email or password");
		}
	};
	const onError = (errors) => {
		console.log("Validation errors:", errors);
	};

	return (
		<div className={cn("flex flex-col gap-6", className)} {...props}>
			<Card>
				<CardHeader>
					<CardTitle>Login to your account</CardTitle>
					<CardDescription>
						Enter your email below to login to your account
					</CardDescription>
				</CardHeader>
				<CardContent>
					<Form {...form}>
						<form
							onSubmit={form.handleSubmit(onSubmit, onError)}
							className="space-y-6"
						>
							{/* Email */}
							<FormField
								control={form.control}
								name="email"
								render={({ field }) => (
									<FormItem>
										<FormLabel>Email</FormLabel>
										<FormControl>
											<Input
												placeholder="m@example.com"
												{...field}
												type="email"
												autoComplete="email"
											/>
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>

							{/* Password */}
							<FormField
								control={form.control}
								name="password"
								render={({ field }) => (
									<FormItem>
										<div className="flex items-center justify-between">
											<FormLabel>Password</FormLabel>
											<Link
												href="/forgot-password"
												className="text-sm text-muted-foreground hover:text-foreground underline-offset-4 hover:underline"
											>
												Forgot your password?
											</Link>
										</div>
										<FormControl>
											<Input
												type="password"
												{...field}
												autoComplete="current-password"
											/>
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>

							<Button
								type="submit"
								className="w-full"
								disabled={pending}
							>
								Login
							</Button>
						</form>
					</Form>
					<div className="mt-4 text-center text-sm">
						Don&apos;t have an account?{" "}
						<Link
							href="/register"
							className="underline underline-offset-4"
						>
							Register
						</Link>
					</div>
				</CardContent>
			</Card>
		</div>
	);
}
