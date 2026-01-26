import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Link, useSearchParams, useNavigate } from "react-router-dom";
import { NetworkBackground } from "@/components/NetworkBackground";
import { useAuth } from "@/context/AuthContext";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { GoogleLogin } from "@react-oauth/google";

const Login = () => {
  const [isLoading, setIsLoading] = useState(false);
  const { loginGoogle, verify } = useAuth();
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token"); // Keep token logic just in case verify link is clicked later?
  const navigate = useNavigate();

  useEffect(() => {
    if (token) {
      handleVerify(token);
    }
  }, [token]);

  const handleVerify = async (t: string) => {
    try {
      setIsLoading(true);
      await verify(t);
      toast.success("Successfully logged in");
    } catch (e) {
      toast.error("Invalid or expired login link");
      navigate("/login");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleSuccess = async (credentialResponse: any) => {
    if (!credentialResponse.credential) {
      toast.error("Google login failed: No credential");
      return;
    }

    setIsLoading(true);
    try {
      await loginGoogle(credentialResponse.credential);
      toast.success("Logged in with Google");
    } catch (e) {
      console.error("[Login] Google error:", e);
      toast.error("Login failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  if (token && isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-foreground" />
        <span className="ml-2">Verifying credentials...</span>
      </div>
    );
  }

  return (
    <div className="relative min-h-screen bg-background overflow-hidden">
      <NetworkBackground />
      <div className="fixed inset-0 network-grid pointer-events-none opacity-40" />

      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center px-6">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-sm"
        >
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 mb-8 justify-center">
            <div className="w-2 h-2 rounded-full bg-foreground" />
            <span className="text-sm font-medium tracking-tight text-foreground">
              SciNets
            </span>
          </Link>

          <h1 className="text-xl font-medium text-foreground mb-2 text-center">
            SciNets Research Access
          </h1>
          <p className="text-sm text-foreground-muted mb-8 text-center">
            Sign in with Google to continue
          </p>

          <div className="flex justify-center">
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={() => toast.error("Google Login Failed")}
              theme="filled_black"
              shape="pill"
            />
          </div>

          {isLoading && (
            <div className="mt-4 flex justify-center text-sm text-muted-foreground">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Signing in...
            </div>
          )}

        </motion.div>
      </div>
    </div>
  );
};
export default Login;
