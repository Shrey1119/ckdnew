"use client";

import { FormEvent, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

type Mode = "login" | "register";

type AuthResponse = {
  token?: string;
  user?: {
    id: number;
    name: string;
    email: string;
  };
  message?: string;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

export default function Home() {
  const [mode, setMode] = useState<Mode>("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const router = useRouter();

  const endpoint = useMemo(
    () =>
      mode === "login"
        ? "/api/auth/login"
        : "/api/auth/register",
    [mode]
  );

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();

    setLoading(true);
    setMessage("");

    try {
      const payload =
        mode === "register"
          ? { name, email, password }
          : { email, password };


      const response = await fetch(
        `${API_BASE}${endpoint}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }
      );


      const data =
        (await response.json()) as AuthResponse;


      if (!response.ok) {
        setMessage(
          data.message || "Authentication failed"
        );
        return;
      }


      if (data.token) {
        localStorage.setItem(
          "auth_token",
          data.token
        );
      }


      if (mode === "register") {
        setMessage(
          "Registration successful. Please login."
        );
        setMode("login");
        setPassword("");
      } else {
        router.push("/dashboard");
      }


    } catch {
      setMessage(
        "Server unavailable. Check backend on port 5000."
      );
    } finally {
      setLoading(false);
    }
  }


  return (
    <main className="auth-shell">

      <div className="auth-card">

        <div className="auth-header">
          <h1>X-GENO GEN</h1>
          <p>Sign in to your clinical account</p>
        </div>


        <div className="auth-tabs">

          <button
            type="button"
            className={mode === "login" ? "is-active" : ""}
            onClick={() => setMode("login")}
          >
            Login
          </button>


          <button
            type="button"
            className={mode === "register" ? "is-active" : ""}
            onClick={() => setMode("register")}
          >
            Register
          </button>

        </div>


        <form
          onSubmit={handleSubmit}
          className="auth-form"
        >

          {mode === "register" && (
            <label>
              Full Name
              <input
                required
                value={name}
                onChange={(e) =>
                  setName(e.target.value)
                }
                placeholder="Your name"
              />
            </label>
          )}


          <label>
            Email
            <input
              required
              type="email"
              value={email}
              onChange={(e) =>
                setEmail(e.target.value)
              }
              placeholder="you@hospital.com"
            />
          </label>


          <label>
            Password
            <input
              required
              minLength={6}
              type="password"
              value={password}
              onChange={(e) =>
                setPassword(e.target.value)
              }
              placeholder="Minimum 6 characters"
            />
          </label>


          <button
            type="submit"
            disabled={loading}
          >
            {loading
              ? "Please wait..."
              : mode === "login"
              ? "Sign In"
              : "Create Account"}
          </button>

        </form>


        {message && (
          <p className="auth-message">
            {message}
          </p>
        )}

      </div>

    </main>
  );
}