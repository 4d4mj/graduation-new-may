// "use client";
import { logout } from "./actions";
import ChatBox from "./TEMP_Chatbox";

export default function Dashboard({ user }) {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-2">
        Hello, {user.first_name}!
      </h1>
      <p className="mb-4">Role: {user.role}</p>
      <ChatBox />
      <form action={logout}>
        <button
          type="submit"
          className="px-4 py-2 bg-red-600 text-white rounded"
        >
          Logout
        </button>
      </form>
    </div>
  );
}
