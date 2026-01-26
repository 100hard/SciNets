import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { useNavigate } from "react-router-dom";

// API Base URL (assuming proxy or same origin for now)
// In Vite dev, we need to ensure proxy is set up or use absolute URL.
// Backend is likely on 8000. Frontend on 5173/8080.
// Ideally, use a relative path /api and configured proxy.
// Ideally, use a relative path /api and configured proxy.
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8005';

import { User } from "../lib/types";

interface AuthContextType {
    user: User | null;
    loading: boolean;
    login: (email: string) => Promise<void>;
    verify: (token: string) => Promise<void>;
    logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();

    useEffect(() => {
        checkUser();
    }, []);

    const checkUser = async () => {
        try {
            const res = await fetch(`${API_URL}/api/auth/me`, { credentials: "include" });
            if (res.ok) {
                const data = await res.json();
                setUser(data);
            } else {
                setUser(null);
            }
        } catch (e) {
            console.error("Auth check failed", e);
            setUser(null);
        } finally {
            setLoading(false);
        }
    };

    const login = async (email: string) => {
        const res = await fetch(`${API_URL}/api/auth/request-link`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email }),
            credentials: "include"
        });
        if (!res.ok) {
            const errText = await res.text();
            console.error("[Auth] Login request failed:", res.status, errText);
            throw new Error(`Login failed: ${res.status} ${errText}`);
        }
    };

    const verify = async (token: string) => {
        const res = await fetch(`${API_URL}/api/auth/verify-link`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ token }),
            credentials: "include"
        });
        if (!res.ok) throw new Error("Verification failed");
        const data = await res.json();
        setUser(data.user);
        navigate("/discovery");
    };

    const logout = async () => {
        await fetch(`${API_URL}/api/auth/logout`, { method: "POST", credentials: "include" });
        setUser(null);
        navigate("/login");
    };

    return (
        <AuthContext.Provider value={{ user, loading, login, verify, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) throw new Error("useAuth must be used within an AuthProvider");
    return context;
};
