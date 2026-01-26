import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";

import { GoogleOAuthProvider } from "@react-oauth/google";

import { config } from "./config";

const clientId = config.GOOGLE_CLIENT_ID;

createRoot(document.getElementById("root")!).render(
    <GoogleOAuthProvider clientId={clientId}>
        <App />
    </GoogleOAuthProvider>
);
