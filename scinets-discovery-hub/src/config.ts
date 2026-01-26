// Centralized Frontend Configuration
// This checks multiple environment variable names to ensure the API URL is picked up in all environments (Vercel, Docker, etc.)

export const config = {
    // Support both VITE_API_URL (standard) and VITE_API_BASE_URL (common convention)
    // Fallback to localhost ONLY if neither is present (for local dev)
    API_URL: import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || "http://localhost:8005",

    // Google Auth
    GOOGLE_CLIENT_ID: import.meta.env.VITE_GOOGLE_CLIENT_ID || "119137837974-4so3iqha7hm13na8uq0mcbpj3qolharv.apps.googleusercontent.com",

    // Feature Flags
    IS_DEV: import.meta.env.DEV,
};

console.log("[Config] Loaded API URL:", config.API_URL); // Debug log to see what's loaded
