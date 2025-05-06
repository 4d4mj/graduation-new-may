// In @auth/login/actions.js
"use server";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import settings from "@/config/settings";

export async function logout() {
  const cookieStore = cookies();
  cookieStore.delete("session");
  cookieStore.delete("refresh_token");
  redirect("/login");
}

export async function sendChat(payload) {
  const cookieStore = cookies();
  const session = cookieStore.get("session")?.value;

  if (!session) {
    throw new Error("Not authenticated");
  }

  try {
    const res = await fetch(`${settings.apiInternalUrl || "http://backend:8000"}/chat/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Cookie": `session=${session}`
      },
      body: JSON.stringify({
        ...payload,
        user_tz: payload.user_tz || Intl.DateTimeFormat().resolvedOptions().timeZone
      })
    });

    if (!res.ok) {
      throw new Error(`Chat request failed with status ${res.status}`);
    }

    return await res.json();
  } catch (error) {
    console.error("Chat error:", error);
    throw error;
  }
}
