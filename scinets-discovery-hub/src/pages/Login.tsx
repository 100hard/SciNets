import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Link, useSearchParams, useNavigate } from "react-router-dom";
import { NetworkBackground } from "@/components/NetworkBackground";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/context/AuthContext";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";

const Login = () => {
  const [email, setEmail] = useState("");
  const [isSent, setIsSent] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { login, verify } = useAuth();
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token");
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
      navigate("/login"); // Clear param
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) return;

    setIsLoading(true);
    try {
      await login(email);
      setIsSent(true);
      toast.success("Magic link sent to your email");
    } catch (e) {
      toast.error("Failed to send magic link");
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
          <Link to="/" className="flex items-center gap-2 mb-8">
            <div className="w-2 h-2 rounded-full bg-foreground" />
            <span className="text-sm font-medium tracking-tight text-foreground">
              SciNets
            </span>
          </Link>

          {!isSent ? (
            <>
              <h1 className="text-xl font-medium text-foreground mb-2">
                Research Preview Login
              </h1>
              <p className="text-sm text-foreground-muted mb-8">
                Login is required due to limited compute resources.
              </p>

              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email" className="text-sm text-foreground-muted">
                    Email address
                  </Label>
                  <Input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="bg-background border-border/50 focus:border-foreground/50"
                    placeholder="researcher@university.edu"
                    disabled={isLoading}
                    autoFocus
                  />
                </div>

                <Button
                  type="submit"
                  className="w-full bg-foreground text-background hover:bg-foreground/90"
                  disabled={isLoading}
                >
                  {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                  Send Magic Link
                </Button>
              </form>
            </>
          ) : (
            <div className="text-center space-y-4">
              <div className="p-4 bg-muted/50 rounded-lg border border-border/50">
                <p className="text-sm text-foreground">
                  We've sent a secure login link to <br />
                  <span className="font-medium">{email}</span>
                </p>
              </div>
              <p className="text-sm text-foreground-muted">
                Click the link in the email to sign in. <br />
                The link expires in 15 minutes.
              </p>
              <Button
                variant="link"
                onClick={() => setIsSent(false)}
                className="text-foreground-muted hover:text-foreground"
              >
                Use a different email
              </Button>
            </div>
          )}

        </motion.div>
      </div>
    </div>
  );
};
export default Login;
