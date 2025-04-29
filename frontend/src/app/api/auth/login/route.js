// app/api/auth/login/route.ts
import { NextResponse } from 'next/server'
import settings from '@/config/settings'

export async function POST(req) {
  const body = await req.json()

  const apiRes = await fetch(`${settings.apiInternalUrl}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    credentials: 'include',
  })

  if (!apiRes.ok) {
    return NextResponse.json(
      await apiRes.json().catch(() => ({})),
      { status: apiRes.status },
    )
  }

  // Build redirect response *before* streaming starts
  const res = NextResponse.redirect(new URL('/dashboard', req.url))

  // Forward every Set-Cookie header exactly as FastAPI sent it
  apiRes.headers
        .getSetCookie()
        .forEach((c) => res.headers.append('Set-Cookie', c))

  return res
}
