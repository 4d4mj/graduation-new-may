export const env = ['SESSION_SECRET'];   // 👈  make it visible to Edge
import { NextResponse } from "next/server";
import { decrypt } from "@lib/session";

const protectedRoutes = ["/c"];
const publicRoutes = ["/auth/login", "/auth/register"];

export async function middleware(req) {
	const path = req.nextUrl.pathname;
	const isProtectedRoute = protectedRoutes.includes(path);
	const isPublicRoute = publicRoutes.includes(path);

	const cookie = req.cookies.get("session")?.value;
	const session = await decrypt(cookie);
	console.log("session", session);

	if (isProtectedRoute && !session?.sub) {
		return NextResponse.redirect(new URL("/login", req.nextUrl));
	}

	if (isPublicRoute && session?.sub) {
		return NextResponse.redirect(new URL("/c", req.nextUrl));
	}

	// attach header so downstream code can skip decrypting again
	const headers = new Headers(req.headers);
	if (session?.sub) headers.set("x-user-id", String(session.sub));

	return NextResponse.next({ request: { headers } });
}
