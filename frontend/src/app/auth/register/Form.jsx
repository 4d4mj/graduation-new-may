"use client";

import * as React from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useFormStatus } from "react-dom";
import { register } from "./actions";
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
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import DatePicker from "@/components/ui/date-picker";
import { registerSchema } from "@/schemas/register";

export function RegisterForm({ className, ...props }) {
	const { pending } = useFormStatus();

	// Initialize the form with react-hook-form and zod validation
	const form = useForm({
		resolver: zodResolver(registerSchema),
		defaultValues: {
			first_name: "",
			last_name: "",
			sex: "M", // Set a default value
			dob: undefined,
			email: "",
			password: "",
			repeat_password: "",
			phone: "",
			address: "",
		},
	});

	// Handle form submission
	const onSubmit = async (data) => {
		const result = await register(data);
		if (result.errors) {
			toast.error(result.errors.api[0] || "Registration failed");
		}
	};

	return (
		<Form {...form}>
			<div className={cn("flex flex-col gap-6", className)} {...props}>
				<Card>
					<CardHeader>
						<CardTitle>Create a new account</CardTitle>
						<CardDescription>
							Fill in your information below to create a new
							account
						</CardDescription>
					</CardHeader>
					<CardContent>
						<form
							onSubmit={form.handleSubmit(onSubmit)}
							className="space-y-6"
						>
							<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
								{/* First Name */}
								<FormField
									control={form.control}
									name="first_name"
									render={({ field }) => (
										<FormItem>
											<FormLabel>First name</FormLabel>
											<FormControl>
												<Input
													placeholder="John"
													{...field}
													autoComplete="given-name"
												/>
											</FormControl>
											<FormMessage />
										</FormItem>
									)}
								/>

								{/* Last Name */}
								<FormField
									control={form.control}
									name="last_name"
									render={({ field }) => (
										<FormItem>
											<FormLabel>Last name</FormLabel>
											<FormControl>
												<Input
													placeholder="Doe"
													{...field}
													autoComplete="family-name"
												/>
											</FormControl>
											<FormMessage />
										</FormItem>
									)}
								/>
							</div>

							{/* Sex */}
							<FormField
								control={form.control}
								name="sex"
								render={({ field }) => (
									<FormItem>
										<FormLabel>Sex</FormLabel>
										<FormControl>
											<RadioGroup
												onValueChange={field.onChange}
												value={field.value}
												className="flex"
											>
												<RadioGroupItem value="M">
													Male
												</RadioGroupItem>
												<RadioGroupItem value="F">
													Female
												</RadioGroupItem>
											</RadioGroup>
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>

							{/* Date of Birth */}
							<FormField
								control={form.control}
								name="dob"
								render={({ field }) => (
									<FormItem className="grid">
										<FormLabel>Date of Birth</FormLabel>
										<DatePicker
											date={field.value}
											setDate={field.onChange}
											startDate={new Date(1900, 1)}
											endDate={new Date()}
										/>
										<FormMessage />
									</FormItem>
								)}
							/>

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

							{/* Phone */}
							<FormField
								control={form.control}
								name="phone"
								render={({ field }) => (
									<FormItem>
										<FormLabel>Phone Number</FormLabel>
										<FormControl>
											<Input
												placeholder="+1 (555) 123-4567"
												{...field}
												type="tel"
												autoComplete="tel"
											/>
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>

							{/* Address */}
							<FormField
								control={form.control}
								name="address"
								render={({ field }) => (
									<FormItem>
										<FormLabel>Address</FormLabel>
										<FormControl>
											<Textarea
												placeholder="123 Main St, City, State, Zip"
												{...field}
												autoComplete="street-address"
											/>
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>

							{/* Password */}
							<div className="flex gap-2">
								<FormField
									control={form.control}
									name="password"
									render={({ field }) => (
										<FormItem>
											<FormLabel>Password</FormLabel>
											<FormControl>
												<Input
													type="password"
													{...field}
													autoComplete="new-password"
												/>
											</FormControl>
											<FormMessage />
										</FormItem>
									)}
								/>
								{/* Repeat Password */}
								<FormField
									control={form.control}
									name="repeat_password"
									render={({ field }) => (
										<FormItem>
											<FormLabel>Repeat Password</FormLabel>
											<FormControl>
												<Input
													type="password"
													{...field}
													autoComplete="new-password"
												/>
											</FormControl>
											<FormMessage />
										</FormItem>
									)}
								/>
							</div>

							<Button
								type="submit"
								className="w-full"
								disabled={pending}
							>
								Register
							</Button>
						</form>
						<div className="mt-4 text-center text-sm">
							Already have an account?{" "}
							<Link
								href="/login"
								className="underline underline-offset-4"
							>
								Login
							</Link>
						</div>
					</CardContent>
				</Card>
			</div>
		</Form>
	);
}
