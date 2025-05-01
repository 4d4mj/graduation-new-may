import "server-only";
import { SignJWT, jwtVerify } from "jose";
import { cookies } from "next/headers";
import settings from "@/config/settings";


// export async function createSession(userId) {
//   const cookieStore = await cookies();
//   const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);
//   const session = await encrypt({ userId, expiresAt });

//   cookieStore.set({
//     name: "session",
//     value: session,
//     httpOnly: true,
//     secure: true,
//     expires: expiresAt,
//     path: "/",
//   });
// }

export async function deleteSession() {
  const cookieStore = await cookies();
  cookieStore.delete("session");
}

// export async function encrypt(payload) {
//   return new SignJWT(payload)
//     .setProtectedHeader({ alg: "HS256" })
//     .setIssuedAt()
//     .setExpirationTime("7d")
//     .sign(encodedKey);
// }

export async function decrypt(token = "") {
  const secret = process.env.SESSION_SECRET;   // <-- edge runtime has it now
  if (!secret) {
    console.error("SESSION_SECRET is missing in Edge runtime");
    return null;
  }

  try {
    const key = new TextEncoder().encode(secret);      // build key *now*
    const { payload } = await jwtVerify(token, key, {
      algorithms: ["HS256"],
    });
    return payload;                                   // { sub, role, exp }
  } catch {
    console.log("Failed to verify session");
    return null;
  }
}


export async function getUser() {
  const cookieStore = await cookies();
  const cookieList = cookieStore.getAll();
  const cookieHeader = cookieList.map(({ name, value }) => `${name}=${value}`).join('; ');
  try {
    const res = await fetch(`${settings.apiInternalUrl}/auth/me`, {
      headers: { Cookie: cookieHeader },
    });
    if (!res.ok) {
      return null;
    }
    return await res.json();
  } catch (error) {
    console.error('Failed to fetch user data', error);
    return null;
  }
}
