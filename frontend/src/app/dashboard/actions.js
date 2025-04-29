// In @auth/login/actions.js
"use server";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";

export async function logout() {
  const cookieStore = cookies();
  cookieStore.delete("session");
  cookieStore.delete("refresh_token");
  redirect("/login");
}
