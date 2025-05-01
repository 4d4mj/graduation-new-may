// lib/user.ts
import { cookies } from "next/headers";
import settings from "@/config/settings";

export async function getUser() {
  const cookieStore = await cookies();            // ① read cookies available to this SSR request
  const cookieHeader = cookieStore          // ② turn them into "key=value; key2=value2"
    .getAll()
    .map(({ name, value }) => `${name}=${value}`)
    .join("; ");

  const res = await fetch(`${settings.apiInternalUrl}/auth/me`, {
    headers: { cookie: cookieHeader },      // ③ forward them to FastAPI
    // Important: DON’T set credentials: "include" in a server call;
    // that flag is for browsers only.
    cache: "no-store",
  });
  if (res.status === 401) console.log(res);

  if (!res.ok) return null;
  return res.json();
}
