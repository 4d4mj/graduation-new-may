// In @auth/login/actions.js
"use server";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import settings from "@/config/settings";

export async function logout() {
  // Properly await the cookies() call
  const cookieStore = await cookies();
  cookieStore.delete("session");
  cookieStore.delete("refresh_token");
  redirect("/login");
}

export async function sendChat(payload) {
  // Properly await the cookies() call
  const cookieStore = await cookies();
  const session = cookieStore.get("session")?.value;

  if (!session) {
    return {
      success: false,
      error: {
        type: "AUTHENTICATION_FAILED",
        status: 401,
        message: "Not authenticated - session token missing"
      }
    };
  }

  try {
    const res = await fetch(`${settings.apiInternalUrl || "http://backend:8000"}/chat/message`, {
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

    // Handle HTTP errors
    if (!res.ok) {
      let errorDetails = { status: res.status };      try {
        const errorData = await res.json();
        // Handle complex error responses (like 422 validation errors)
        if (typeof errorData.detail === 'object' && errorData.detail !== null) {
          if (errorData.detail.message) {
            // For validation errors with structured format
            errorDetails.message = errorData.detail.message;
            if (errorData.detail.errors && Array.isArray(errorData.detail.errors)) {
              // Append validation error details
              const errorMessages = errorData.detail.errors.map(err => err.message).join(', ');
              errorDetails.message += ` (${errorMessages})`;
            }
          } else {
            // Fallback for other complex objects
            errorDetails.message = JSON.stringify(errorData.detail);
          }
        } else {
          // Simple string detail
          errorDetails.message = errorData.detail || errorData.message;
        }
      } catch {
        // Failed to parse error response
        errorDetails.message = `Request failed with status ${res.status}`;
      }

      // Handle specific status codes
      switch (res.status) {
        case 401:
          // Token expired or invalid - clear cookies
          cookieStore.delete("session");
          cookieStore.delete("refresh_token");
          errorDetails.message = "Session expired. Please log in again.";
          return {
            success: false,
            error: {
              type: "AUTHENTICATION_FAILED",
              status: 401,
              message: errorDetails.message
            }
          };

        case 403:
          errorDetails.message = errorDetails.message || "Access denied. Insufficient permissions.";
          return {
            success: false,
            error: {
              type: "FORBIDDEN",
              status: 403,
              message: errorDetails.message
            }
          };

        case 422:
          errorDetails.message = errorDetails.message || "Invalid request data.";
          return {
            success: false,
            error: {
              type: "VALIDATION_ERROR",
              status: 422,
              message: errorDetails.message
            }
          };

        case 500:
          errorDetails.message = errorDetails.message || "Server error occurred.";
          return {
            success: false,
            error: {
              type: "SERVER_ERROR",
              status: 500,
              message: errorDetails.message
            }
          };

        default:
          return {
            success: false,
            error: {
              type: "HTTP_ERROR",
              status: res.status,
              message: errorDetails.message
            }
          };
      }
    }

    const data = await res.json();
    return {
      success: true,
      data: data
    };

  } catch (error) {
    console.error("Chat error:", error);

    // Return network/fetch errors instead of throwing
    if (error.name === "TypeError" && error.message.includes("fetch")) {
      return {
        success: false,
        error: {
          type: "NETWORK_ERROR",
          status: 503,
          message: "Unable to connect to server"
        }
      };
    }

    return {
      success: false,
      error: {
        type: "UNKNOWN_ERROR",
        status: 500,
        message: error.message || "An unexpected error occurred"
      }
    };
  }
}
